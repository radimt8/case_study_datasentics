import torch
import numpy as np
from typing import Dict, List, Optional
from .bayesian_mcmc import BayesianPMF_MCMC


class MCMCRecommender:
    """Recommendation system using MCMC samples"""
    
    def __init__(self, model: BayesianPMF_MCMC, book_titles: List[str]):
        self.model = model
        self.book_titles = book_titles
        self.samples = None
        self.training_results = None
    
    def train(self, R_normalized: torch.Tensor, mask: torch.Tensor,
              R_test_normalized: Optional[torch.Tensor] = None,
              mask_test: Optional[torch.Tensor] = None,
              n_samples: int = 1500, 
              burn_in: int = 500,
              use_map_init: bool = True,
              adaptive: bool = True,
              **kwargs) -> Dict:
        """
        Train the MCMC model with optional adaptive sampling
        """
        import time
        start_time = time.time()
        
        print(f"Training MCMC model (adaptive={adaptive}, MAP init={use_map_init})...")
        
        # Run Gibbs sampling
        results = self.model.gibbs_sample(
            R_normalized, mask,
            R_test=R_test_normalized,
            mask_test=mask_test,
            n_samples=n_samples,
            burn_in=burn_in,
            use_map_init=use_map_init,
            check_convergence=adaptive,
            verbose=True,
            **kwargs
        )
        
        self.samples = results['samples']
        self.training_results = results
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'n_samples': results['n_samples'],
            'n_iterations': results['n_iterations'],
            'burn_in': burn_in,
            'converged': results.get('converged', False),
            'used_map_init': use_map_init
        }
    
    def mcmc_recommendations_with_uncertainty(self, user_idx: int, 
                                             observed_mask: torch.Tensor,
                                             top_k: int = 10) -> Dict:
        """Generate recommendations with uncertainty from MCMC samples"""
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Stack samples
        U_samples = torch.stack(self.samples['U'])  # (n_samples, n_users, k)
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        
        # Get unrated items
        user_mask = observed_mask[user_idx]
        unrated_items = (~user_mask).nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(unrated_items) == 0:
            return {"error": "User has rated all items"}
        
        # Convert to tensor for indexing
        unrated_items_tensor = torch.from_numpy(unrated_items).long()
        
        # Predictions across all samples
        user_factors = U_samples[:, user_idx]  # (n_samples, k)
        item_factors = V_samples[:, unrated_items_tensor]  # (n_samples, n_unrated, k)
        
        raw_preds = torch.einsum('sk,snk->sn', user_factors, item_factors)
        sigmoid_preds = torch.sigmoid(raw_preds)
        
        # Statistics
        pred_means = sigmoid_preds.mean(dim=0)
        pred_stds = sigmoid_preds.std(dim=0)
        
        # Top-k items
        sorted_indices = torch.argsort(pred_means, descending=True)[:top_k]
        recommended_items = unrated_items[sorted_indices.cpu().numpy()]
        book_titles = [self.book_titles[i] for i in recommended_items]
        
        # Scale to 1-10
        scaled_means = (pred_means[sorted_indices] * 9 + 1).tolist()
        
        # Credible intervals
        quantiles = torch.quantile(sigmoid_preds[:, sorted_indices],
                                  torch.tensor([0.025, 0.975]), dim=0)
        lower_ci = torch.clamp(quantiles[0] * 9 + 1, 1, 10).tolist()
        upper_ci = torch.clamp(quantiles[1] * 9 + 1, 1, 10).tolist()
        
        # Uncertainty as CI half-width
        ci_half_widths = [(upper - lower) / 2 for lower, upper in zip(lower_ci, upper_ci)]
        
        return {
            'recommendations': [
                {
                    'title': title,
                    'predicted_rating': round(rating, 2),
                    'uncertainty': round(half_width, 2),
                    'confidence_interval': [round(lower, 2), round(upper, 2)]
                }
                for title, rating, half_width, lower, upper in zip(
                    book_titles, scaled_means, ci_half_widths, lower_ci, upper_ci
                )
            ]
        }
    
    def evaluate_model(self, R_test: torch.Tensor, mask_test: torch.Tensor) -> Dict:
        """Evaluate model on test set"""
        if self.samples is None:
            raise ValueError("Model not trained yet.")
        
        # Use posterior mean for evaluation
        U_mean = torch.stack(self.samples['U']).mean(dim=0)
        V_mean = torch.stack(self.samples['V']).mean(dim=0)
        
        # Predictions
        predictions = torch.sigmoid(U_mean @ V_mean.T)
        
        # Test metrics
        test_indices = mask_test.nonzero(as_tuple=True)
        test_preds = predictions[test_indices]
        test_actual = R_test[test_indices]
        
        mse = torch.mean((test_preds - test_actual) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(test_preds - test_actual))
        
        return {
            'rmse': rmse.item(),
            'mae': mae.item(),
            'rmse_original_scale': rmse.item() * 9,
            'mae_original_scale': mae.item() * 9,
            'n_test_ratings': len(test_preds)
        }
    
    def generate_all_recommendations(self, observed_mask: torch.Tensor,
                                    top_k: int = 10) -> Dict:
        """Generate recommendations for all users"""
        if self.samples is None:
            raise ValueError("Model not trained yet.")
        
        all_recommendations = {}
        n_users = observed_mask.shape[0]
        
        print(f"Generating recommendations for {n_users} users...")
        
        for user_idx in range(n_users):
            if user_idx % 100 == 0:
                print(f"Processing user {user_idx}/{n_users}")
            
            try:
                recs = self.mcmc_recommendations_with_uncertainty(
                    user_idx, observed_mask, top_k
                )
                
                if 'error' not in recs:
                    all_recommendations[user_idx] = recs
                    
            except Exception as e:
                print(f"Error for user {user_idx}: {e}")
                continue
        
        print(f"Generated recommendations for {len(all_recommendations)} users")
        return all_recommendations
