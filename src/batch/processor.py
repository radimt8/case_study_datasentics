import json
import redis
import pickle
import os
from datetime import datetime
from typing import Dict

from ..utils.data_processor import BookDataProcessor
from ..models.bayesian_mcmc import BayesianPMF_MCMC
from ..models.mcmc_recommender import MCMCRecommender


class BatchProcessor:
    """Batch processing service for MCMC model training and precomputation"""
    
    def __init__(self, redis_url: str = None, data_path: str = "./data"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.data_path = data_path
        
        self.data_processor = BookDataProcessor(data_path)
        self.model = None
        self.recommender = None
        
    def run_full_pipeline(self, min_book_ratings: int = 50,
                         min_user_ratings: int = 20,
                         k: int = 20,
                         n_samples: int = 1500,
                         burn_in: int = 100,
                         alpha: float = 2.0,
                         top_k_recs: int = 10,
                         use_adaptive: bool = True) -> Dict:
        """Run the complete MCMC training pipeline"""
        
        # Validation
        if n_samples < 100:
            print(f"⚠️  Warning: n_samples={n_samples} is very low. Recommendations may be poor.")
            print("   Recommended minimum: 200 for dev, 500 for staging, 1500 for production")
        
        if burn_in < 10:
            burn_in = 10
            print(f"⚠️  Burn-in too low, setting to minimum of {burn_in}")
        
        print("=" * 60)
        print(f"ADAPTIVE MCMC BATCH PROCESSING STARTED: {datetime.now()}")
        print("=" * 60)
        
        # Step 1: Data preprocessing
        print("\n1. PREPROCESSING DATA...")
        filtered_data = self.data_processor.preprocess_data(
            min_book_ratings=min_book_ratings,
            min_user_ratings=min_user_ratings
        )
        
        # Step 2: Create rating matrices
        print("\n2. CREATING RATING MATRICES...")
        matrices = self.data_processor.create_rating_matrix(filtered_data)
        
        # Step 3: Create MCMC model
        print(f"\n3. CREATING BAYESIAN MCMC MODEL (k={k})...")
        self.model = BayesianPMF_MCMC(
            n_users=matrices['n_users'],
            n_items=matrices['n_items'],
            k=k,
            alpha=alpha
        )
        
        # Step 4: Create recommender
        print("\n4. CREATING MCMC RECOMMENDATION SERVICE...")
        self.recommender = MCMCRecommender(
            model=self.model,
            book_titles=matrices['book_titles']
        )
        
        # Step 5: Train with adaptive MCMC
        print(f"\n5. TRAINING WITH ADAPTIVE MCMC...")
        print(f"   - MAP initialization: ENABLED")
        print(f"   - Adaptive stopping: {'ENABLED' if use_adaptive else 'DISABLED'}")
        print(f"   - Max samples: {n_samples}, Burn-in: {burn_in}")
        
        # Calculate minimum samples for convergence checking
        # Need at least 100 samples for reliable convergence metrics
        min_samples_for_convergence = min(100, n_samples // 2)
        
        training_results = self.recommender.train(
            matrices['R_normalized'],
            matrices['mask_tensor'],
            R_test_normalized=matrices.get('R_test_normalized'),
            mask_test=matrices.get('mask_test_tensor'),
            n_samples=n_samples,
            burn_in=burn_in,
            use_map_init=False,
            adaptive=use_adaptive,
            min_samples=min_samples_for_convergence,  # Dynamic minimum
            check_every=max(10, n_samples // 20)      # Dynamic check frequency
        )
        
        print(f"\nTraining completed in {training_results['training_time']:.1f}s")
        print(f"Collected {training_results['n_samples']} samples")
        if training_results.get('converged'):
            print("✓ Model converged!")
        
        # Step 6: Evaluate model
        print("\n6. EVALUATING MODEL PERFORMANCE...")
        evaluation_metrics = self.recommender.evaluate_model(
            matrices['R_test_normalized'],
            matrices['mask_test_tensor']
        )
        
        print(f"Test RMSE (1-10 scale): {evaluation_metrics['rmse_original_scale']}")
        print(f"Test MAE (1-10 scale): {evaluation_metrics['mae_original_scale']}")
        
        # Step 7: Generate all recommendations (trained users)
        print("\n7. GENERATING RECOMMENDATIONS FOR TRAINED USERS...")
        trained_recommendations = self.recommender.generate_all_recommendations(
            matrices['mask_tensor'],
            top_k=top_k_recs
        )
        
        # Step 7b: Generate cold start recommendations for filtered users
        print("\n7b. GENERATING COLD START RECOMMENDATIONS FOR FILTERED USERS...")
        cold_start_recommendations = self._generate_cold_start_recommendations(
            filtered_data, top_k_recs
        )
        
        # Combine both recommendation sets
        all_recommendations = {**trained_recommendations, **cold_start_recommendations}
        
        # Step 8: Store in Redis/Dragonfly
        print("\n8. STORING RECOMMENDATIONS IN DRAGONFLY...")
        stored_count = self._store_recommendations_in_redis(
            all_recommendations, matrices
        )
        
        # Step 9: Store model metadata and components
        print("\n9. STORING MODEL METADATA AND COMPONENTS...")
        self._store_model_metadata(training_results, evaluation_metrics, matrices, 
                                 k, n_samples, burn_in)
        
        # Step 10: Store recommender model and book titles for cold start
        print("\n10. STORING MODEL COMPONENTS FOR COLD START...")
        self._store_model_components(matrices)
        
        # Return comprehensive results
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results,
            'evaluation_metrics': evaluation_metrics,
            'dataset_stats': {
                'n_users': matrices['n_users'],
                'n_items': matrices['n_items'],
                'n_ratings_train': matrices['mask_tensor'].sum().item(),
                'n_ratings_test': matrices['mask_test_tensor'].sum().item()
            },
            'recommendations_stored': stored_count,
            'model_parameters': {
                'k': k,
                'n_samples': n_samples,
                'actual_samples_collected': training_results['n_samples'],
                'burn_in': burn_in,
                'alpha': alpha,
                'min_book_ratings': min_book_ratings,
                'min_user_ratings': min_user_ratings
            },
            'performance': {
                'converged': training_results.get('converged', False),
                'training_time_seconds': training_results['training_time'],
                'test_rmse': evaluation_metrics['rmse_original_scale'],
                'test_mae': evaluation_metrics['mae_original_scale']
            }
        }
        
        return pipeline_results
    
    def _store_recommendations_in_redis(self, all_recommendations: Dict, 
                                      matrices: Dict) -> int:
        """Store recommendations in Redis/Dragonfly"""
        
        stored_count = 0
        
        # Create pipeline for efficient Redis operations
        pipe = self.redis_client.pipeline()
        
        for key, rec_data in all_recommendations.items():
            if isinstance(key, str) and key.startswith('cold_start_'):
                # Cold start user - extract user ID from key
                user_id = key.replace('cold_start_', '')
                redis_key = f"recs:{user_id}"
                value = json.dumps(rec_data['recommendations'])
                pipe.set(redis_key, value)
                stored_count += 1
            elif isinstance(key, int):
                # Trained user - convert internal index to external user ID
                external_user_id = self.data_processor.get_external_user_id(key)
                
                if external_user_id != -1:
                    # Store recommendations for this user
                    redis_key = f"recs:{external_user_id}"
                    value = json.dumps(rec_data['recommendations'])
                    pipe.set(redis_key, value)
                    stored_count += 1
        
        # Execute all Redis operations
        pipe.execute()
        
        return stored_count
    
    def _store_model_metadata(self, training_results: Dict, 
                             evaluation_metrics: Dict, matrices: Dict,
                             k: int, n_samples: int, burn_in: int):
        """Store model metadata and statistics"""
        
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'model_type': 'MCMC',
            'model_performance': evaluation_metrics,
            'training_info': {
                'training_time': training_results['training_time'],
                'n_samples_collected': training_results['n_samples'],  # Actual samples collected
                'n_iterations': training_results['n_iterations'],      # Total iterations run
                'burn_in': training_results.get('burn_in', burn_in),
                'converged': training_results.get('converged', False),
                'used_map_init': training_results.get('used_map_init', False)
            },
            'dataset_info': {
                'n_users': matrices['n_users'],
                'n_items': matrices['n_items'],
                'sparsity': 1 - (matrices['mask_tensor'].sum().item() / 
                               (matrices['n_users'] * matrices['n_items']))
            },
            'model_params': {
                'k': k,
                'requested_samples': n_samples,  # What was requested
                'actual_samples': training_results['n_samples'],  # What we got
                'burn_in': burn_in,
                'alpha': 2.0
            }
        }
        
        # Store metadata
        self.redis_client.set('model:metadata', json.dumps(metadata))
        
        # Store user and book mappings for API
        self.redis_client.set('model:user_count', matrices['n_users'])
        self.redis_client.set('model:book_count', matrices['n_items'])
    
    def _store_model_components(self, matrices: Dict):
        """Store model components for cold start and search functionality"""
        try:
            # Store the trained recommender model
            recommender_pickle = pickle.dumps(self.recommender).decode('latin1')
            self.redis_client.set('model:recommender', recommender_pickle)
            
            # Store book titles for search functionality
            book_titles_json = json.dumps(matrices['book_titles'])
            self.redis_client.set('model:book_titles', book_titles_json)
            
            print(f"✓ Stored recommender model and {len(matrices['book_titles'])} book titles")
            
        except Exception as e:
            print(f"⚠️  Failed to store model components: {e}")
            # Don't fail the entire pipeline for this
    
    def _generate_cold_start_recommendations(self, filtered_data, top_k_recs: int) -> Dict:
        """Generate cold start recommendations for users filtered out during preprocessing"""
        cold_start_recs = {}
        
        try:
            # Load original data and apply same preprocessing as training data
            users, ratings, books = self.data_processor.load_data()
            
            # Clean and merge original data (same as preprocessing)
            ratings_clean = ratings[['User-ID', 'ISBN', 'Book-Rating']]
            books_clean = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']]
            original_ratings = ratings_clean.merge(books_clean, on='ISBN', how='inner')
            
            # Filter to explicit ratings only (same as preprocessing)
            explicit_only = original_ratings[original_ratings['Book-Rating'] > 0]
            
            # Get the same book filtering as used in training
            # Only include books that are in our final trained model
            trained_books = set(filtered_data['Book-Title'].unique())
            explicit_only = explicit_only[explicit_only['Book-Title'].isin(trained_books)]
            
            print(f"After filtering to trained books: {len(explicit_only)} ratings")
            print(f"Unique users in filtered data: {explicit_only['User-ID'].nunique()}")
            
            # Get users that have ratings for trained books but were filtered out
            users_with_trained_book_ratings = set(explicit_only['User-ID'].unique())
            trained_users = set(filtered_data['User-ID'].unique())
            filtered_out_users = users_with_trained_book_ratings - trained_users
            
            print(f"Total users who rated trained books: {len(users_with_trained_book_ratings)}")
            print(f"Trained users: {len(trained_users)}")
            print(f"Cold start users to process: {len(filtered_out_users)}")
            print(f"Expected total coverage: {len(users_with_trained_book_ratings)} users")
            
            processed_count = 0
            for user_id in filtered_out_users:
                if processed_count % 100 == 0 and processed_count > 0:
                    print(f"Processed {processed_count}/{len(filtered_out_users)} cold start users")
                
                try:
                    # Get user's ratings for books in our trained model
                    user_ratings = explicit_only[explicit_only['User-ID'] == user_id]
                    
                    # Convert to format expected by cold start method
                    user_ratings_list = []
                    for _, rating_row in user_ratings.iterrows():
                        book_title = rating_row['Book-Title']
                        if book_title in self.recommender.book_title_to_idx:
                            user_ratings_list.append({
                                'title': book_title,
                                'rating': float(rating_row['Book-Rating'])
                            })
                    
                    # Only process if user has ratings for books in our model
                    if len(user_ratings_list) > 0:
                        result = self.recommender.cold_start_recommendations(
                            user_ratings_list, top_k_recs
                        )
                        
                        if 'error' not in result:
                            # Store using internal index (negative to distinguish from trained users)
                            # We'll use the external user ID directly in storage
                            cold_start_recs[f"cold_start_{user_id}"] = result
                            processed_count += 1
                
                except Exception as e:
                    print(f"Error processing cold start user {user_id}: {e}")
                    continue
            
            print(f"Generated cold start recommendations for {processed_count} users")
            return cold_start_recs
            
        except Exception as e:
            print(f"⚠️  Failed to generate cold start recommendations: {e}")
            return {}
    
    def health_check(self) -> Dict:
        """Check system health"""
        try:
            # Test Redis connection
            self.redis_client.ping()
            redis_status = "connected"
            
            # Check if model metadata exists
            metadata = self.redis_client.get('model:metadata')
            model_status = "available" if metadata else "not_trained"
            
            # Count stored recommendations
            rec_keys = self.redis_client.keys('recs:*')
            rec_count = len(rec_keys)
            
            return {
                'status': 'healthy',
                'redis_status': redis_status,
                'model_status': model_status,
                'recommendations_count': rec_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main entry point for batch processing"""
    import os
    
    processor = BatchProcessor()
    
    # Load hyperparameters from environment variables with fallbacks
    def get_env_float(key, default):
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
            
    def get_env_int(key, default):
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
            
    def get_env_bool(key, default):
        val = os.getenv(key, str(default)).lower()
        return val in ('true', '1', 'yes', 'on')
    
    # Environment-based configuration with .env override
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        # Production settings - high quality
        base_config = {
            'min_book_ratings': 20,
            'min_user_ratings': 10,
            'k': 5,
            'n_samples': 1500,
            'burn_in': 200,
            'alpha': 3.0,
            'top_k_recs': 20,
            'use_adaptive': True
        }
    elif env == 'staging':
        # Staging - medium quality, faster
        base_config = {
            'min_book_ratings': 20,
            'min_user_ratings': 10,
            'k': 5,
            'n_samples': 550,
            'burn_in': 200,
            'alpha': 6.57,
            'top_k_recs': 15,
            'use_adaptive': True
        }
    else:
        # Development - fast iteration
        base_config = {
            'min_book_ratings': 20,
            'min_user_ratings': 10,
            'k': 5,
            'n_samples': 50,
            'burn_in': 5,
            'alpha': 3.0,
            'top_k_recs': 10,
            'use_adaptive': True
        }
    
    # Override with environment variables if provided
    config = {
        'min_book_ratings': get_env_int('MCMC_MIN_BOOK_RATINGS', base_config['min_book_ratings']),
        'min_user_ratings': get_env_int('MCMC_MIN_USER_RATINGS', base_config['min_user_ratings']),
        'k': get_env_int('MCMC_K', base_config['k']),
        'n_samples': get_env_int('MCMC_N_SAMPLES', base_config['n_samples']),
        'burn_in': get_env_int('MCMC_BURN_IN', base_config['burn_in']),
        'alpha': get_env_float('MCMC_ALPHA', base_config['alpha']),
        'top_k_recs': get_env_int('MCMC_TOP_K_RECS', base_config['top_k_recs']),
        'use_adaptive': get_env_bool('MCMC_USE_ADAPTIVE', base_config['use_adaptive'])
    }
    
    print(f"Running in {env} mode with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        results = processor.run_full_pipeline(**config)
        
        print("\nMCMC Pipeline completed successfully!")
        print(f"✓ Collected {results['training_results']['n_samples']} samples")
        print(f"✓ Test RMSE: {results['evaluation_metrics']['rmse_original_scale']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"MCMC Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
