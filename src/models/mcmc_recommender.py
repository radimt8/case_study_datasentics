import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .bayesian_mcmc import BayesianPMF_MCMC


class MCMCRecommender:
    """Recommendation system using MCMC samples for uncertainty quantification"""
    
    def __init__(self, model: BayesianPMF_MCMC, book_titles: List[str]):
        self.model = model
        self.book_titles = book_titles
        self.samples = None
    
    def train(self, R_normalized: torch.Tensor, mask: torch.Tensor, 
              n_samples: int = 1500, burn_in: int = 500, **kwargs) -> Dict:
        """Train the MCMC model"""
        print(f"Training MCMC with {n_samples} samples, {burn_in} burn-in...")
        
        import time
        start_time = time.time()
        
        # Run MCMC sampling
        self.samples = self.model.gibbs_sample(
            R_normalized, mask, n_samples=n_samples, burn_in=burn_in, verbose=True
        )
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'n_samples': n_samples,
            'burn_in': burn_in,
            'total_iterations': n_samples + burn_in
        }
    
    def mcmc_recommendations_with_uncertainty(self, user_idx: int, observed_mask: torch.Tensor, 
                                            top_k: int = 10) -> Dict:
        """Generate recommendations directly from MCMC samples"""
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Stack samples for vectorized computation
        U_samples = torch.stack(self.samples['U'])  # (n_samples, n_users, k)
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        
        # Get user's unrated items - FIX HERE
        user_mask = observed_mask[user_idx]  # This is a tensor
        unrated_items = (~user_mask).nonzero(as_tuple=True)[0].cpu().numpy()  # Convert to numpy here
        
        if len(unrated_items) == 0:
            return {"error": "User has rated all items"}
        
        # Convert back to tensor for indexing
        unrated_items_tensor = torch.from_numpy(unrated_items).long()
        
        # Vectorized prediction across all MCMC samples
        user_factors = U_samples[:, user_idx]  # (n_samples, k)
        item_factors = V_samples[:, unrated_items_tensor]  # (n_samples, n_unrated, k)
        
        # Compute predictions for all samples at once
        raw_preds = torch.einsum('sk,snk->sn', user_factors, item_factors)
        sigmoid_preds = torch.sigmoid(raw_preds)
        
        # Compute statistics
        pred_means = sigmoid_preds.mean(dim=0)
        pred_stds = sigmoid_preds.std(dim=0)
        
        # Sort by prediction mean
        sorted_indices = torch.argsort(pred_means, descending=True)[:top_k]
        
        # Get book titles and scale to 1-10
        recommended_items = unrated_items[sorted_indices.cpu().numpy()]
        book_titles = [self.book_titles[i] for i in recommended_items]
        
        scaled_means = (pred_means[sorted_indices] * 9 + 1).tolist()
        scaled_stds = (pred_stds[sorted_indices] * 9).tolist()
        
        # Compute credible intervals
        quantiles = torch.quantile(sigmoid_preds[:, sorted_indices], 
                                  torch.tensor([0.025, 0.975]), dim=0)
        lower_ci = torch.clamp(quantiles[0] * 9 + 1, 1, 10).tolist()
        upper_ci = torch.clamp(quantiles[1] * 9 + 1, 1, 10).tolist()
        
        # Calculate uncertainty as actual half-width of CI (more intuitive)
        ci_half_widths = [(upper - lower) / 2 for lower, upper in zip(lower_ci, upper_ci)]
        
        return {
            'recommendations': [
                {
                    'title': title,
                    'predicted_rating': round(rating, 2),
                    'uncertainty': round(half_width, 2),  # Now matches CI half-width
                    'confidence_interval': [round(lower, 2), round(upper, 2)]
                }
                for title, rating, half_width, lower, upper in zip(
                    book_titles, scaled_means, ci_half_widths, lower_ci, upper_ci
                )
            ]
        }
    
    def generate_all_recommendations(self, observed_mask: torch.Tensor, 
                                   top_k: int = 10, n_samples: int = 500) -> Dict:
        """Generate recommendations for all users"""
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
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
                print(f"Error generating recommendations for user {user_idx}: {e}")
                continue
        
        print(f"Generated recommendations for {len(all_recommendations)} users")
        return all_recommendations
    
    def evaluate_model(self, R_test: torch.Tensor, mask_test: torch.Tensor) -> Dict:
        """Evaluate model performance on test set"""
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use mean of samples for evaluation
        U_mean = torch.stack(self.samples['U']).mean(dim=0)
        V_mean = torch.stack(self.samples['V']).mean(dim=0)
        
        # Compute predictions
        predictions = torch.sigmoid(U_mean @ V_mean.T)
        
        # Get test predictions and actual values
        test_indices = mask_test.nonzero(as_tuple=True)
        test_preds = predictions[test_indices]
        test_actual = R_test[test_indices]
        
        # Compute metrics
        mse = torch.mean((test_preds - test_actual) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(test_preds - test_actual))
        
        return {
            'rmse': rmse.item(),
            'mae': mae.item(),
            'rmse_original_scale': rmse.item() * 9,  # Convert back to 1-10 scale
            'mae_original_scale': mae.item() * 9,
            'n_test_ratings': len(test_preds)
        }
