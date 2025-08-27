import json
import redis
import pickle
import os
from datetime import datetime
from typing import Dict

from ..utils.data_processor import BookDataProcessor
from ..models.bayesian_mf import BayesianMatrixFactorization
from ..models.recommender import BayesianRecommender


class BatchProcessor:
    """Batch processing service for model training and precomputation"""
    
    def __init__(self, redis_url: str = None, data_path: str = "./data"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.data_path = data_path
        
        self.data_processor = BookDataProcessor(data_path)
        self.model = None
        self.recommender = None
        
    def run_full_pipeline(self, min_book_ratings: int = 50, 
                         min_user_ratings: int = 20,
                         k: int = 5,
                         max_epochs: int = 500,
                         learning_rate: float = 0.001,
                         top_k_recs: int = 10,
                         n_samples: int = 500) -> Dict:
        """Run the complete training and precomputation pipeline"""
        
        print("=" * 60)
        print(f"BATCH PROCESSING STARTED: {datetime.now()}")
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
        
        # Step 3: Train model
        print("\n3. TRAINING BAYESIAN MATRIX FACTORIZATION...")
        self.model = BayesianMatrixFactorization(
            n_users=matrices['n_users'],
            n_items=matrices['n_items'],
            k=k
        )
        
        training_results = self.model.train(
            matrices['R_normalized'],
            matrices['mask_tensor'],
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            verbose=True
        )
        
        # Step 4: Create recommender
        print("\n4. CREATING RECOMMENDATION SERVICE...")
        self.recommender = BayesianRecommender(
            model=self.model,
            book_titles=matrices['book_titles']
        )
        
        # Step 5: Evaluate model
        print("\n5. EVALUATING MODEL PERFORMANCE...")
        evaluation_metrics = self.recommender.evaluate_model(
            matrices['R_test_normalized'],
            matrices['mask_test_tensor']
        )
        
        print(f"Test RMSE (1-10 scale): {evaluation_metrics['rmse_original_scale']}")
        print(f"Test MAE (1-10 scale): {evaluation_metrics['mae_original_scale']}")
        
        # Step 6: Generate all recommendations
        print("\n6. GENERATING RECOMMENDATIONS FOR ALL USERS...")
        all_recommendations = self.recommender.generate_all_recommendations(
            matrices['mask_tensor'],
            top_k=top_k_recs,
            n_samples=n_samples
        )
        
        # Step 7: Store in Redis/Dragonfly
        print("\n7. STORING RECOMMENDATIONS IN DRAGONFLY...")
        stored_count = self._store_recommendations_in_redis(
            all_recommendations, matrices
        )
        
        # Step 8: Store model metadata
        print("\n8. STORING MODEL METADATA...")
        self._store_model_metadata(training_results, evaluation_metrics, matrices)
        
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
                'min_book_ratings': min_book_ratings,
                'min_user_ratings': min_user_ratings,
                'max_epochs': max_epochs,
                'learning_rate': learning_rate
            }
        }
        
        print("\n" + "=" * 60)
        print(f"BATCH PROCESSING COMPLETED: {datetime.now()}")
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
                            evaluation_metrics: Dict, matrices: Dict):
        """Store model metadata and statistics"""
        
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'model_performance': evaluation_metrics,
            'training_info': {
                'final_elbo': training_results['final_elbo'],
                'converged_epoch': training_results['converged_epoch'],
                'total_epochs': len(training_results['elbo_history'])
            },
            'dataset_info': {
                'n_users': matrices['n_users'],
                'n_items': matrices['n_items'],
                'sparsity': 1 - (matrices['mask_tensor'].sum().item() / 
                               (matrices['n_users'] * matrices['n_items']))
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
            k=5,
            max_epochs=500,
            learning_rate=0.001,
            top_k_recs=10,
            n_samples=500
        )
        
        print("\nPipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()