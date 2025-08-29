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
                         k: int = 10,
                         n_samples: int = 1500,
                         burn_in: int = 500,
                         alpha: float = 2.0,
                         top_k_recs: int = 10) -> Dict:
        """Run the complete MCMC training and precomputation pipeline"""
        
        print("=" * 60)
        print(f"MCMC BATCH PROCESSING STARTED: {datetime.now()}")
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
        
        # Step 4: Create MCMC recommender
        print("\n4. CREATING MCMC RECOMMENDATION SERVICE...")
        self.recommender = MCMCRecommender(
            model=self.model,
            book_titles=matrices['book_titles']
        )
        
        # Step 5: Train with MCMC sampling
        print(f"\n5. TRAINING WITH MCMC ({n_samples} samples, {burn_in} burn-in)...")
        training_results = self.recommender.train(
            matrices['R_normalized'],
            matrices['mask_tensor'],
            n_samples=n_samples,
            burn_in=burn_in
        )
        
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
                'burn_in': burn_in,
                'alpha': alpha,
                'min_book_ratings': min_book_ratings,
                'min_user_ratings': min_user_ratings
            }
        }
        
        print("\n" + "=" * 60)
        print(f"MCMC BATCH PROCESSING COMPLETED: {datetime.now()}")
        print(f"Recommendations stored for {stored_count} users")
        print("=" * 60)
        
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
                'n_samples': training_results['n_samples'],
                'burn_in': training_results['burn_in'],
                'total_iterations': training_results['total_iterations']
            },
            'dataset_info': {
                'n_users': matrices['n_users'],
                'n_items': matrices['n_items'],
                'sparsity': 1 - (matrices['mask_tensor'].sum().item() / 
                               (matrices['n_users'] * matrices['n_items']))
            },
            'model_params': {
                'k': k,
                'n_samples': n_samples,
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
    processor = BatchProcessor()
    
    try:
        results = processor.run_full_pipeline(
            min_book_ratings=50,
            min_user_ratings=20,
            k=10,
            n_samples=1500,
            burn_in=500,
            alpha=2.0,
            top_k_recs=10
        )
        
        print("\nMCMC Pipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"MCMC Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
