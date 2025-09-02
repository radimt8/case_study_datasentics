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
        self.book_title_to_idx = {title: idx for idx, title in enumerate(book_titles)}
    
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
    
    def mcmc_recommendations_with_uncertainty(self, user_factors: torch.Tensor, 
                                             unrated_items: torch.Tensor,
                                             top_k: int = 10) -> Dict:
        """Generate recommendations with uncertainty from MCMC samples
        
        Args:
            user_factors: (n_samples, k) tensor of user factor samples
            unrated_items: tensor of item indices to consider for recommendations
            top_k: number of recommendations to return
        """
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get V samples
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        
        if len(unrated_items) == 0:
            return {"error": "User has rated all items"}
        
        # Get item factors for unrated items
        item_factors = V_samples[:, unrated_items]  # (n_samples, n_unrated, k)
        
        raw_preds = torch.einsum('sk,snk->sn', user_factors, item_factors)
        sigmoid_preds = torch.sigmoid(raw_preds * 2) # Multiply by 2 for steeper sigmoid
        
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
    
    def create_cold_start_user_tensors(self, original_ratings_data, filtered_user_ids) -> torch.Tensor:
        """Create user factor tensors for cold-start users
        
        Args:
            original_ratings_data: DataFrame with original ratings
            filtered_user_ids: List of user IDs that were filtered out
            
        Returns:
            U_cold: (n_samples, n_cold_users, k) tensor
        """
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        n_samples, n_items, k = V_samples.shape
        
        cold_users_list = []
        valid_cold_users = []
        
        for user_id in filtered_user_ids:
            # Get user's ratings
            user_ratings = original_ratings_data[original_ratings_data['User-ID'] == user_id]
            
            # Convert to format expected by construction logic
            valid_ratings = []
            for _, rating_row in user_ratings.iterrows():
                book_title = rating_row['Book-Title']
                if book_title in self.book_title_to_idx:
                    valid_ratings.append({
                        'title': book_title,
                        'rating': float(rating_row['Book-Rating'])
                    })
            
            if len(valid_ratings) > 0:
                # Create user vector for each V sample
                user_vectors_across_samples = []
                
                for sample_idx in range(n_samples):
                    V_sample = V_samples[sample_idx]  # (n_items, k)
                    
                    # Build user vector from weighted average
                    user_vector = torch.zeros(k)
                    total_weight = 0
                    
                    for rating_info in valid_ratings:
                        book_idx = self.book_title_to_idx[rating_info['title']]
                        rating = float(rating_info['rating'])
                        
                        user_vector += rating * V_sample[book_idx]
                        total_weight += rating
                    
                    # Normalize
                    if total_weight > 0:
                        user_vector = user_vector / total_weight
                    
                    user_vectors_across_samples.append(user_vector)
                
                # Stack across samples
                user_tensor = torch.stack(user_vectors_across_samples)  # (n_samples, k)
                cold_users_list.append(user_tensor)
                valid_cold_users.append(user_id)
        
        if len(cold_users_list) > 0:
            # Stack across users: (n_samples, n_cold_users, k)
            U_cold = torch.stack(cold_users_list, dim=1)
            return U_cold, valid_cold_users
        else:
            return torch.empty(n_samples, 0, k), []

    def generate_all_recommendations(self, observed_mask: torch.Tensor,
                                    original_ratings_data=None, 
                                    filtered_user_ids=None,
                                    top_k: int = 10) -> Dict:
        """Generate recommendations for all users including cold-start users"""
        if self.samples is None:
            raise ValueError("Model not trained yet.")
        
        # Get trained user samples
        U_trained = torch.stack(self.samples['U'])  # (n_samples, n_trained, k)
        n_samples, n_trained, k = U_trained.shape
        
        all_recommendations = {}
        
        # Add cold-start users if provided
        if original_ratings_data is not None and filtered_user_ids is not None:
            print("Creating cold-start user tensors...")
            U_cold, valid_cold_users = self.create_cold_start_user_tensors(
                original_ratings_data, filtered_user_ids
            )
            
            # Concatenate trained and cold-start users
            U_all = torch.cat([U_trained, U_cold], dim=1)  # (n_samples, n_total, k)
            user_id_mapping = list(range(n_trained)) + valid_cold_users
            
            print(f"Processing {n_trained} trained + {len(valid_cold_users)} cold-start users...")
        else:
            U_all = U_trained
            user_id_mapping = list(range(n_trained))
            print(f"Processing {n_trained} trained users...")
        
        n_total = U_all.shape[1]
        
        # Process all users
        for user_idx in range(n_total):
            if user_idx % 100 == 0:
                print(f"Processing user {user_idx}/{n_total}")
            
            try:
                user_factors = U_all[:, user_idx]  # (n_samples, k)
                
                # Determine unrated items
                if user_idx < n_trained:
                    # Trained user - use observed_mask
                    user_mask = observed_mask[user_idx]
                    unrated_items = (~user_mask).nonzero(as_tuple=True)[0]
                else:
                    # Cold-start user - assume all items are unrated
                    unrated_items = torch.arange(self.samples['V'][0].shape[0])
                
                recs = self.mcmc_recommendations_with_uncertainty(
                    user_factors, unrated_items, top_k
                )
                
                if 'error' not in recs:
                    actual_user_id = user_id_mapping[user_idx] 
                    all_recommendations[actual_user_id] = recs
                    
            except Exception as e:
                print(f"Error for user {user_idx}: {e}")
                continue
        
        print(f"Generated recommendations for {len(all_recommendations)} users")
        return all_recommendations
    
    def cold_start_filtered_user(self, user_id: int, original_ratings_data, top_k: int = 10) -> Dict:
        """Generate recommendations for user filtered out during preprocessing
        
        Args:
            user_id: Original user ID from dataset
            original_ratings_data: DataFrame with original ratings data
            top_k: Number of recommendations to return
        """
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get user's ratings from original data
        user_ratings = original_ratings_data[original_ratings_data['User-ID'] == user_id]
        
        if len(user_ratings) == 0:
            return {"error": f"No ratings found for user {user_id} in original dataset"}
        
        # Filter to books that exist in our trained model
        valid_ratings = []
        for _, rating_row in user_ratings.iterrows():
            book_title = rating_row['Book-Title']
            if book_title in self.book_title_to_idx:
                valid_ratings.append({
                    'title': book_title,
                    'rating': float(rating_row['Book-Rating'])
                })
        
        if len(valid_ratings) == 0:
            return {"error": f"User {user_id} has no ratings for books in our trained model"}
        
        # Use the same cold start method as before
        result = self.cold_start_recommendations(valid_ratings, top_k)
        
        # Add user info
        if 'error' not in result:
            result['original_user_id'] = user_id
            result['total_original_ratings'] = len(user_ratings)
        
        return result
    
    def cold_start_recommendations(self, user_ratings: List[Dict], top_k: int = 10) -> Dict:
        """Generate recommendations for new user using MCMC uncertainty
        
        Args:
            user_ratings: List of dicts with 'title' and 'rating' keys
            top_k: Number of recommendations to return
        """
        if self.samples is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        n_samples, n_items, k = V_samples.shape
        
        # Build user vectors across all samples (for uncertainty)
        user_factors_list = []
        valid_ratings = 0
        
        # Find valid books first
        valid_book_info = []
        for rating_info in user_ratings:
            if rating_info['title'] in self.book_title_to_idx:
                valid_book_info.append({
                    'idx': self.book_title_to_idx[rating_info['title']],
                    'rating': float(rating_info['rating']),
                    'title': rating_info['title']
                })
                valid_ratings += 1
        
        if valid_ratings == 0:
            return {"error": "No rated books found in dataset"}
        
        # Create user vector for each V sample
        for sample_idx in range(n_samples):
            V_sample = V_samples[sample_idx]  # (n_items, k)
            
            user_vector = torch.zeros(k)
            total_weight = 0
            
            for book_info in valid_book_info:
                book_idx = book_info['idx']
                rating = book_info['rating']
                
                user_vector += rating * V_sample[book_idx]
                total_weight += rating
            
            # Normalize
            if total_weight > 0:
                user_vector = user_vector / total_weight
            
            user_factors_list.append(user_vector)
        
        # Stack user factors across samples: (n_samples, k)
        user_factors = torch.stack(user_factors_list)
        
        # DEBUG: Check user factor variation across samples
        # print(f"DEBUG User Factors - shape: {user_factors.shape}")
        # print(f"DEBUG User Factors - mean across samples: {user_factors.mean(dim=0).tolist()}")
        # print(f"DEBUG User Factors - std across samples: {user_factors.std(dim=0).tolist()}")
        # print(f"DEBUG User Factors - norm per sample: {user_factors.norm(dim=1)[:5].tolist()}")
        # print(f"DEBUG User Factors - sample 0 vs 1 diff: {(user_factors[0] - user_factors[1]).norm().item():.6f}")
        
        # DEBUG: Compare with a trained user
        U_trained = torch.stack(self.samples['U'])  # (n_samples, n_users, k)
        V_samples = torch.stack(self.samples['V'])  # (n_samples, n_items, k)
        
        # Calculate scale ratio between U and V
        U_scale = U_trained.norm(dim=2).mean().item()  # Mean norm across all users and samples
        V_scale = V_samples.norm(dim=2).mean().item()  # Mean norm across all items and samples
        scale_ratio = U_scale / V_scale
        
        print(f"DEBUG Scales - U mean norm: {U_scale:.6f}, V mean norm: {V_scale:.6f}")
        print(f"DEBUG Scale ratio (U/V): {scale_ratio:.2f}")
        print(f"DEBUG Cold-start norm before scaling: {user_factors.norm(dim=1).mean().item():.6f}")
        
        # Apply scale correction
        user_factors = user_factors * scale_ratio
        
        # print(f"DEBUG Cold-start norm after scaling: {user_factors.norm(dim=1).mean().item():.6f}")
        
        trained_user_0 = U_trained[:, 0]  # (n_samples, k) - first trained user
        # print(f"DEBUG Trained User 0 - norm per sample: {trained_user_0.norm(dim=1)[:5].tolist()}")
        # print(f"DEBUG Corrected ratio: {trained_user_0.norm(dim=1).mean().item() / user_factors.norm(dim=1).mean().item():.2f}")
        
        # Get all unrated items (exclude rated books)
        rated_book_indices = [book_info['idx'] for book_info in valid_book_info]
        all_items = torch.arange(n_items)
        unrated_items = all_items[~torch.isin(all_items, torch.tensor(rated_book_indices))]
        
        # Use the refactored MCMC method
        result = self.mcmc_recommendations_with_uncertainty(user_factors, unrated_items, top_k)
        
        # DEBUG: Analyze recommended books vs input books
        if 'error' not in result and 'recommendations' in result:
            V_mean = V_samples.mean(dim=0)  # (n_items, k)
            user_vector_mean = user_factors.mean(dim=0)  # (k,)
            
            print(f"\nDEBUG BOOK ANALYSIS:")
            print(f"Input books and their factors:")
            for book_info in valid_book_info:
                book_factors = V_mean[book_info['idx']]
                similarity = torch.cosine_similarity(user_vector_mean, book_factors, dim=0)
                print(f"  {book_info['title']} (rating {book_info['rating']}): "
                    f"cosine_sim={similarity.item():.3f}, norm={book_factors.norm().item():.3f}")
            
            print(f"\nRecommended books and their similarity to user vector:")
            for i, rec in enumerate(result['recommendations'][:3]):  # Show top 3
                book_idx = self.book_title_to_idx[rec['title']]
                book_factors = V_mean[book_idx]
                similarity = torch.cosine_similarity(user_vector_mean, book_factors, dim=0)
                print(f"  #{i+1}: {rec['title']} (pred {rec['predicted_rating']}): "
                    f"cosine_sim={similarity.item():.3f}, norm={book_factors.norm().item():.3f}")
        
        # Add metadata
        if 'error' not in result:
            result['user_vector_norm'] = user_factors.mean(dim=0).norm().item()
            result['valid_books_rated'] = valid_ratings
        
        return result
