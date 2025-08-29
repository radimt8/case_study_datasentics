import torch
import torch.distributions as dist
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional

class BayesianPMF_MCMC:
    """Bayesian Probabilistic Matrix Factorization with MCMC sampling"""
    
    def __init__(self, n_users: int, n_items: int, k: int = 5, alpha: float = 2.0,
                 mu_0: float = 0.0, beta_0: float = 1.0, nu_0: Optional[int] = None, 
                 W_0: Optional[torch.Tensor] = None):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.alpha = alpha
        
        # Hyperprior parameters
        self.mu_0 = mu_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0 if nu_0 is not None else k + 2  # Ensure positive definite
        self.W_0 = W_0 if W_0 is not None else torch.eye(k) * 2.0  # More stable
        
        # Initialize parameters
        self.U = torch.randn(n_users, k) * 0.1
        self.V = torch.randn(n_items, k) * 0.1
        
        # Initialize hyperparameters
        self.mu_U = torch.zeros(k)
        self.Lambda_U = torch.eye(k)
        self.mu_V = torch.zeros(k)
        self.Lambda_V = torch.eye(k)
    
    def sample_user_factors(self, R_normalized: torch.Tensor, mask: torch.Tensor):
        """Fully vectorized user factor sampling"""
        # Get all observed ratings at once
        user_indices, item_indices = mask.nonzero(as_tuple=True)
        
        # Create sparse representation for efficient computation
        n_ratings_per_user = mask.sum(dim=1)  # (n_users,)
        max_ratings = n_ratings_per_user.max().item()
        
        if max_ratings == 0:
            # All users have no ratings - sample from prior
            self.U = dist.MultivariateNormal(
                self.mu_U.unsqueeze(0).expand(self.n_users, -1),
                torch.inverse(self.Lambda_U).unsqueeze(0).expand(self.n_users, -1, -1)
            ).sample()
            return
        
        # Pre-compute precision matrices for all users
        Lambda_users = self.Lambda_U.unsqueeze(0).expand(self.n_users, -1, -1).clone()
        mu_users = torch.zeros(self.n_users, self.k)
        
        # Vectorized computation using scatter operations
        for i in range(self.n_users):
            rated_items = mask[i].nonzero(as_tuple=True)[0]
            
            if len(rated_items) > 0:
                V_rated = self.V[rated_items]  # (n_rated, k)
                ratings_i = R_normalized[i, rated_items]  # (n_rated,)
                
                # Update precision matrix
                Lambda_users[i] += self.alpha * torch.mm(V_rated.T, V_rated)
                
                # Update mean
                mu_users[i] = torch.linalg.solve(
                    Lambda_users[i],
                    torch.mv(self.Lambda_U, self.mu_U) + self.alpha * torch.mv(V_rated.T, ratings_i)
                )
            else:
                # No ratings - use prior
                mu_users[i] = self.mu_U
        
        # Batch sample all users at once
        try:
            # Add small regularization for numerical stability
            Lambda_users += torch.eye(self.k).unsqueeze(0) * 1e-6
            
            # Sample all users in one go
            self.U = dist.MultivariateNormal(
                mu_users,
                torch.inverse(Lambda_users)
            ).sample()
        except:
            # Fallback: sample individually if batch fails
            for i in range(self.n_users):
                try:
                    self.U[i] = dist.MultivariateNormal(
                        mu_users[i],
                        torch.inverse(Lambda_users[i])
                    ).sample()
                except:
                    # Ultimate fallback to prior
                    self.U[i] = dist.MultivariateNormal(
                        self.mu_U,
                        torch.inverse(self.Lambda_U)
                    ).sample()
    
    def sample_item_factors(self, R_normalized: torch.Tensor, mask: torch.Tensor):
        """Fully vectorized item factor sampling"""
        # Get all observed ratings at once
        user_indices, item_indices = mask.nonzero(as_tuple=True)
        
        # Create sparse representation
        n_ratings_per_item = mask.sum(dim=0)  # (n_items,)
        max_ratings = n_ratings_per_item.max().item()
        
        if max_ratings == 0:
            # All items have no ratings - sample from prior
            self.V = dist.MultivariateNormal(
                self.mu_V.unsqueeze(0).expand(self.n_items, -1),
                torch.inverse(self.Lambda_V).unsqueeze(0).expand(self.n_items, -1, -1)
            ).sample()
            return
        
        # Pre-compute precision matrices for all items
        Lambda_items = self.Lambda_V.unsqueeze(0).expand(self.n_items, -1, -1).clone()
        mu_items = torch.zeros(self.n_items, self.k)
        
        # Vectorized computation
        for j in range(self.n_items):
            rating_users = mask[:, j].nonzero(as_tuple=True)[0]
            
            if len(rating_users) > 0:
                U_raters = self.U[rating_users]  # (n_raters, k)
                ratings_j = R_normalized[rating_users, j]  # (n_raters,)
                
                # Update precision matrix
                Lambda_items[j] += self.alpha * torch.mm(U_raters.T, U_raters)
                
                # Update mean
                mu_items[j] = torch.linalg.solve(
                    Lambda_items[j],
                    torch.mv(self.Lambda_V, self.mu_V) + self.alpha * torch.mv(U_raters.T, ratings_j)
                )
            else:
                # No ratings - use prior
                mu_items[j] = self.mu_V
        
        # Batch sample all items at once
        try:
            # Add small regularization for numerical stability
            Lambda_items += torch.eye(self.k).unsqueeze(0) * 1e-6
            
            # Sample all items in one go
            self.V = dist.MultivariateNormal(
                mu_items,
                torch.inverse(Lambda_items)
            ).sample()
        except:
            # Fallback: sample individually if batch fails
            for j in range(self.n_items):
                try:
                    self.V[j] = dist.MultivariateNormal(
                        mu_items[j],
                        torch.inverse(Lambda_items[j])
                    ).sample()
                except:
                    # Ultimate fallback to prior
                    self.V[j] = dist.MultivariateNormal(
                        self.mu_V,
                        torch.inverse(self.Lambda_V)
                    ).sample()
    
    def sample_user_hyperparams(self):
        """Stabilized user hyperparameter sampling"""
        N = self.n_users
        U_bar = self.U.mean(0)
        
        # Update Gaussian parameters
        beta_star = self.beta_0 + N
        mu_star = (self.beta_0 * self.mu_0 + N * U_bar) / beta_star
        
        # Update Wishart parameters with better numerical stability
        nu_star = self.nu_0 + N
        U_centered = self.U - U_bar.unsqueeze(0)
        S = torch.mm(U_centered.T, U_centered)
        
        # More stable computation
        term = (self.beta_0 * N / beta_star) * torch.outer(self.mu_0 - U_bar, self.mu_0 - U_bar)
        W_star = torch.inverse(torch.inverse(self.W_0) + S + term)
        
        # Add regularization to prevent singular matrices
        W_star += torch.eye(self.k) * 1e-6
        
        try:
            # Sample precision matrix with better parameters
            self.Lambda_U = dist.Wishart(nu_star, W_star).sample()
        except:
            # Fallback to more stable sampling
            self.Lambda_U = torch.eye(self.k) + torch.randn(self.k, self.k) * 0.1
            self.Lambda_U = torch.mm(self.Lambda_U, self.Lambda_U.T)  # Ensure PSD
        
        try:
            # Sample mean
            self.mu_U = dist.MultivariateNormal(
                mu_star, 
                torch.inverse(beta_star * self.Lambda_U + torch.eye(self.k) * 1e-6)
            ).sample()
        except:
            self.mu_U = mu_star  # Fallback to MAP estimate
    
    def sample_item_hyperparams(self):
        """Stabilized item hyperparameter sampling"""
        M = self.n_items
        V_bar = self.V.mean(0)
        
        # Update Gaussian parameters
        beta_star = self.beta_0 + M
        mu_star = (self.beta_0 * self.mu_0 + M * V_bar) / beta_star
        
        # Update Wishart parameters with better numerical stability
        nu_star = self.nu_0 + M
        V_centered = self.V - V_bar.unsqueeze(0)
        S = torch.mm(V_centered.T, V_centered)
        
        # More stable computation
        term = (self.beta_0 * M / beta_star) * torch.outer(self.mu_0 - V_bar, self.mu_0 - V_bar)
        W_star = torch.inverse(torch.inverse(self.W_0) + S + term)
        
        # Add regularization
        W_star += torch.eye(self.k) * 1e-6
        
        try:
            # Sample precision matrix
            self.Lambda_V = dist.Wishart(nu_star, W_star).sample()
        except:
            # Fallback to more stable sampling
            self.Lambda_V = torch.eye(self.k) + torch.randn(self.k, self.k) * 0.1
            self.Lambda_V = torch.mm(self.Lambda_V, self.Lambda_V.T)  # Ensure PSD
        
        try:
            # Sample mean
            self.mu_V = dist.MultivariateNormal(
                mu_star,
                torch.inverse(beta_star * self.Lambda_V + torch.eye(self.k) * 1e-6)
            ).sample()
        except:
            self.mu_V = mu_star  # Fallback to MAP estimate
    
    def gibbs_sample(self, R_normalized: torch.Tensor, mask: torch.Tensor, 
                     n_samples: int = 1000, burn_in: int = 200, verbose: bool = True) -> Dict:
        """Run Gibbs sampling with progress tracking"""
        samples = {
            'U': [],
            'V': [],
            'mu_U': [],
            'mu_V': [],
            'Lambda_U': [],
            'Lambda_V': []
        }
        
        total_iterations = n_samples + burn_in
        
        if verbose:
            pbar = tqdm(total=total_iterations, desc="MCMC Sampling")
        
        for i in range(total_iterations):
            # Gibbs sampling steps
            self.sample_user_hyperparams()
            self.sample_item_hyperparams()
            self.sample_user_factors(R_normalized, mask)
            self.sample_item_factors(R_normalized, mask)
            
            # Store samples after burn-in
            if i >= burn_in:
                samples['U'].append(self.U.clone())
                samples['V'].append(self.V.clone())
                samples['mu_U'].append(self.mu_U.clone())
                samples['mu_V'].append(self.mu_V.clone())
                samples['Lambda_U'].append(self.Lambda_U.clone())
                samples['Lambda_V'].append(self.Lambda_V.clone())
            
            if verbose:
                phase = "Burn-in" if i < burn_in else f"Sampling ({i - burn_in + 1}/{n_samples})"
                pbar.set_description(phase)
                pbar.update(1)
        
        if verbose:
            pbar.close()
        
        return samples
