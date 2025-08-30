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
        
        # Step 7: Generate all recommendations
        print("\n7. GENERATING RECOMMENDATIONS FOR ALL USERS...")
        all_recommendations = self.recommender.generate_all_recommendations(
            matrices['mask_tensor'],
            top_k=top_k_recs
        )
        
        # Step 8: Store in Redis/Dragonfly
        print("\n8. STORING RECOMMENDATIONS IN DRAGONFLY...")
        stored_count = self._store_recommendations_in_redis(
            all_recommendations, matrices
        )
        
        # Step 9: Store model metadata
        print("\n9. STORING MODEL METADATA...")
        self._store_model_metadata(training_results, evaluation_metrics, matrices, 
                                 k, n_samples, burn_in)
        
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
        
        for internal_idx, rec_data in all_recommendations.items():
            # Convert internal index to external user ID
            external_user_id = self.data_processor.get_external_user_id(internal_idx)
            
            if external_user_id != -1:
                # Store recommendations for this user
                key = f"recs:{external_user_id}"
                value = json.dumps(rec_data['recommendations'])
                pipe.set(key, value)
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
    
    # Environment-based configuration
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        # Production settings - high quality
        config = {
            'min_book_ratings': 50,
            'min_user_ratings': 20,
            'k': 30,
            'n_samples': 1500,  # More samples for production
            'burn_in': 200,     # Longer burn-in
            'alpha': 1.0,
            'top_k_recs': 20,   # More recommendations stored
            'use_adaptive': True
        }
    elif env == 'staging':
        # Staging - medium quality, faster
        config = {
            'min_book_ratings': 50,
            'min_user_ratings': 20,
            'k': 30,
            'n_samples': 500,
            'burn_in': 100,
            'alpha': 1.0,
            'top_k_recs': 15,
            'use_adaptive': True
        }
    else:
        # Development - fast iteration
        config = {
            'min_book_ratings': 50,
            'min_user_ratings': 20,
            'k': 30,
            'n_samples': 50,   # Minimum for decent quality
            'burn_in': 5,      # Reduced with MAP init
            'alpha': 1.0,
            'top_k_recs': 10,
            'use_adaptive': True
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
