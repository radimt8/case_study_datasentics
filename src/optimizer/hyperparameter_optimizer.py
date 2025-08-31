import os
import numpy as np
from typing import Dict, List, Tuple
import optuna
from optuna.samplers import TPESampler
import torch

from ..batch.processor import BatchProcessor
from ..utils.data_processor import BookDataProcessor
from ..models.bayesian_mcmc import BayesianPMF_MCMC
from ..models.mcmc_recommender import MCMCRecommender


class HyperparameterOptimizer:
    """Bayesian optimization for MCMC hyperparameters"""
    
    def __init__(self, data_path: str = "./data", redis_url: str = None):
        self.data_path = data_path
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Load and preprocess data once
        print("Loading and preprocessing data...")
        self.data_processor = BookDataProcessor(data_path)
        self.filtered_data = self.data_processor.preprocess_data(
            min_book_ratings=50,
            min_user_ratings=20
        )
        self.matrices = self.data_processor.create_rating_matrix(self.filtered_data)
        print(f"Data loaded: {self.matrices['n_users']} users, {self.matrices['n_items']} items")
        
    def objective_function(self, trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to maximize (Optuna maximizes by default with direction='maximize')
        """
        # Sample hyperparameters
        alpha = trial.suggest_float('alpha', 1.0, 15.0)
        k = trial.suggest_int('k', 5, 25)
        reg_log = trial.suggest_float('regularization_log', -10, -6)
        n_samples = trial.suggest_int('n_samples', 200, 600)
        burn_in = trial.suggest_int('burn_in', 50, 200)
        
        regularization = 10 ** reg_log  # Convert from log scale
        
        print(f"\n--- Trying: alpha={alpha:.2f}, k={k}, reg={regularization:.2e}, "
              f"samples={n_samples}, burn_in={burn_in} ---")
        
        try:
            # Create and train model
            model = BayesianPMF_MCMC(
                n_users=self.matrices['n_users'],
                n_items=self.matrices['n_items'],
                k=k,
                alpha=alpha
            )
            
            # Temporarily modify regularization in the model
            original_reg = None
            if hasattr(model, '_regularization'):
                original_reg = model._regularization
            model._regularization = regularization
            
            recommender = MCMCRecommender(
                model=model,
                book_titles=self.matrices['book_titles']
            )
            
            # Train with optimization parameters
            training_results = recommender.train(
                self.matrices['R_normalized'],
                self.matrices['mask_tensor'],
                R_test_normalized=self.matrices.get('R_test_normalized'),
                mask_test=self.matrices.get('mask_test_tensor'),
                n_samples=n_samples,
                burn_in=burn_in,
                use_map_init=False,
                adaptive=True
            )
            
            # Calculate user factor diversity (our main metric)
            U_samples = torch.stack(recommender.samples['U'])
            U_mean = U_samples.mean(dim=0)  # (n_users, k)
            
            # User variance across the latent space
            user_variance = U_mean.var(dim=0).mean().item()
            
            # User-to-user differences
            user_diffs = []
            n_comparisons = min(100, self.matrices['n_users'])  # Sample for efficiency
            indices = np.random.choice(self.matrices['n_users'], n_comparisons, replace=False)
            
            for i in range(len(indices)-1):
                diff = (U_mean[indices[i]] - U_mean[indices[i+1]]).norm().item()
                user_diffs.append(diff)
            
            avg_user_diff = np.mean(user_diffs)
            
            # Evaluation metrics
            eval_metrics = recommender.evaluate_model(
                self.matrices['R_test_normalized'],
                self.matrices['mask_test_tensor']
            )
            
            # Composite score (higher is better)
            # We want high user diversity and low RMSE
            diversity_score = user_variance * 10 + avg_user_diff  # Scale up diversity
            accuracy_score = max(0, 2.0 - eval_metrics['rmse_original_scale'])  # Penalty for high RMSE
            
            total_score = diversity_score + accuracy_score * 0.5  # Weight diversity higher
            
            print(f"Results: user_var={user_variance:.3f}, avg_diff={avg_user_diff:.3f}, "
                  f"RMSE={eval_metrics['rmse_original_scale']:.3f}, score={total_score:.3f}")
            
            # Return negative because Optuna minimizes and we want to maximize score
            return -total_score
            
        except Exception as e:
            print(f"Error: {e}")
            return 1000  # High penalty for failed runs
        finally:
            # Restore original regularization if it was modified
            if original_reg is not None:
                model._regularization = original_reg
    
    def optimize(self, n_calls: int = 20) -> Dict:
        """
        Run Bayesian optimization using Optuna
        
        Args:
            n_calls: Number of optimization iterations
            
        Returns:
            Dictionary with optimal parameters
        """
        print(f"Starting Bayesian optimization with {n_calls} calls...")
        
        # Create study with TPE sampler
        study = optuna.create_study(
            direction='minimize',  # We return negative scores, so minimize
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective_function, n_trials=n_calls)
        
        # Extract optimal parameters
        best_params = study.best_params
        optimal_regularization = 10 ** best_params['regularization_log']
        
        optimal_params = {
            'alpha': best_params['alpha'],
            'k': int(best_params['k']),
            'regularization': optimal_regularization,
            'n_samples': int(best_params['n_samples']),
            'burn_in': int(best_params['burn_in']),
            'score': -study.best_value  # Convert back to positive
        }
        
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print(f"Best score: {-study.best_value:.3f}")
        print(f"Optimal parameters:")
        for key, value in optimal_params.items():
            if key != 'score':
                print(f"  {key}: {value}")
                
        return optimal_params
    
    def update_env_file(self, params: Dict, env_file_path: str = ".env"):
        """Update .env file with optimal parameters"""
        import re
        from datetime import datetime
        
        # Define the MCMC parameters to update
        mcmc_params = {
            'MCMC_ALPHA': f"{params['alpha']:.2f}",
            'MCMC_K': str(params['k']),
            'MCMC_N_SAMPLES': str(params['n_samples']),
            'MCMC_BURN_IN': str(params['burn_in']),
        }
        
        try:
            # Read existing .env file if it exists
            existing_lines = []
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r') as f:
                    existing_lines = f.readlines()
            
            # Track which parameters we've updated
            updated_params = set()
            new_lines = []
            
            # Process existing lines
            for line in existing_lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    new_lines.append(line)
                    continue
                    
                # Check if this line contains an MCMC parameter
                updated = False
                for param_name, param_value in mcmc_params.items():
                    if line.startswith(f"{param_name}="):
                        new_lines.append(f"{param_name}={param_value}")
                        updated_params.add(param_name)
                        updated = True
                        print(f"Updated: {param_name}={param_value}")
                        break
                
                if not updated:
                    new_lines.append(line)
            
            # Add any parameters that weren't found in the existing file
            if updated_params != set(mcmc_params.keys()):
                new_lines.append("")
                new_lines.append(f"# MCMC parameters optimized on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                for param_name, param_value in mcmc_params.items():
                    if param_name not in updated_params:
                        new_lines.append(f"{param_name}={param_value}")
                        print(f"Added: {param_name}={param_value}")
                
                new_lines.append(f"# Optimization score: {params['score']:.3f}")
                new_lines.append(f"# Regularization: {params['regularization']:.2e}")
            
            # Write updated .env file
            with open(env_file_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            
            print(f"\n✅ Successfully updated {env_file_path}")
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to update {env_file_path}: {e}")
            print(f"\n📝 MANUAL COPY-PASTE NEEDED:")
            for param_name, param_value in mcmc_params.items():
                print(f"{param_name}={param_value}")
            return False


def main():
    """Main optimization script"""
    optimizer = HyperparameterOptimizer()
    
    # Run optimization
    n_calls = int(os.getenv('OPTIMIZER_N_CALLS', '15'))
    optimal_params = optimizer.optimize(n_calls=n_calls)
    
    # Update .env file automatically
    success = optimizer.update_env_file(optimal_params)
    
    if success:
        print(f"\n🚀 Ready to run training with optimized parameters!")
        print(f"Next step: docker compose run --rm batch")
    
    return optimal_params


if __name__ == "__main__":
    main()