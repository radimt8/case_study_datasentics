import torch
import torch.distributions as dist
import torch.optim as optim
from scipy import stats
import numpy as np
from collections import deque
from tqdm import tqdm
from typing import Dict, Tuple, Optional


class ConvergenceMonitor:
    """Monitor MCMC convergence using multiple diagnostics"""
    
    def __init__(self, patience: int = 100, window_size: int = 50):
        self.patience = patience
        self.window_size = window_size
        self.metrics_history = {
            'log_likelihood': deque(maxlen=window_size * 2),
            'rmse': deque(maxlen=window_size * 2),
            'param_norm_U': deque(maxlen=window_size * 2),
            'param_norm_V': deque(maxlen=window_size * 2)
        }
        self.best_rmse = float('inf')
        self.patience_counter = 0
        
    def compute_log_likelihood(self, U: torch.Tensor, V: torch.Tensor, 
                              R: torch.Tensor, mask: torch.Tensor, 
                              alpha: float = 2.0) -> float:
        """Compute log-likelihood of the data"""
        pred = torch.sigmoid(U @ V.T)
        mse = ((pred - R) * mask).pow(2).sum()
        log_lik = -0.5 * alpha * mse
        return log_lik.item()
    
    def check_convergence(self, U: torch.Tensor, V: torch.Tensor, 
                         R: torch.Tensor, mask: torch.Tensor,
                         R_test: Optional[torch.Tensor] = None, 
                         mask_test: Optional[torch.Tensor] = None) -> Tuple[bool, Dict]:
        """Check multiple convergence criteria"""
        
        # 1. Log-likelihood
        log_lik = self.compute_log_likelihood(U, V, R, mask)
        self.metrics_history['log_likelihood'].append(log_lik)
        
        # 2. Parameter norms
        u_norm = torch.norm(U).item()
        v_norm = torch.norm(V).item()
        self.metrics_history['param_norm_U'].append(u_norm)
        self.metrics_history['param_norm_V'].append(v_norm)
        
        # 3. Test RMSE if available
        rmse_test = None
        if R_test is not None and mask_test is not None:
            pred_test = torch.sigmoid(U @ V.T)
            rmse_test = torch.sqrt(
                ((pred_test - R_test) * mask_test).pow(2).sum() / mask_test.sum()
            ).item()
            self.metrics_history['rmse'].append(rmse_test)
            
            # Update patience counter
            if rmse_test < self.best_rmse:
                self.best_rmse = rmse_test
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        diagnostics = {
            'log_likelihood': log_lik,
            'param_norm_U': u_norm,
            'param_norm_V': v_norm,
            'rmse_test': rmse_test,
            'patience_counter': self.patience_counter
        }
        
        # Need enough samples before checking convergence
        if len(self.metrics_history['log_likelihood']) < self.window_size * 2:
            return False, diagnostics
        
        # 4. Geweke diagnostic
        recent = list(self.metrics_history['log_likelihood'])[-self.window_size:]
        older = list(self.metrics_history['log_likelihood'])[-2*self.window_size:-self.window_size]
        
        geweke_z = abs(np.mean(recent) - np.mean(older)) / \
                   np.sqrt(np.var(recent)/len(recent) + np.var(older)/len(older))
        diagnostics['geweke_z'] = geweke_z
        
        # 5. Parameter stability
        recent_u = list(self.metrics_history['param_norm_U'])[-self.window_size:]
        rel_change_u = np.std(recent_u) / np.mean(recent_u) if np.mean(recent_u) > 0 else 0
        diagnostics['param_stability'] = rel_change_u
        
        # Convergence criteria
        converged = (
            geweke_z < 2.0 and  # Geweke test
            rel_change_u < 0.01  # Parameter stability
        ) or (
            self.patience_counter >= self.patience  # Early stopping
        )
        
        if self.patience_counter >= self.patience:
            diagnostics['early_stopped'] = True
        
        return converged, diagnostics


class BayesianPMF_MCMC:
    """Bayesian PMF with MCMC following Salakhutdinov & Mnih 2008"""
    
    def __init__(self, n_users: int, n_items: int, k: int = 5, 
                 alpha: float = 2.0, beta_0: float = 2.0):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.alpha = alpha  # Observation precision
        
        # Hyperprior parameters (Section 3.1 of paper)
        self.mu_0 = torch.zeros(k)
        self.beta_0 = beta_0
        self.nu_0 = k  # Degrees of freedom
        self.W_0 = torch.eye(k)  # Scale matrix
        
        # Initialize parameters with small random values
        self.U = torch.randn(n_users, k) * 0.01
        self.V = torch.randn(n_items, k) * 0.01
        
        # Initialize hyperparameters
        self.mu_U = torch.zeros(k)
        self.Lambda_U = torch.eye(k) * 2.0
        self.mu_V = torch.zeros(k)
        self.Lambda_V = torch.eye(k) * 2.0
        
        self.map_trained = False
    
    def train_map(self, R: torch.Tensor, mask: torch.Tensor, 
                  n_epochs: int = 200, lr: float = 0.005,  # Reduced LR
                  lambda_reg: float = 0.001, verbose: bool = True) -> Dict:  # Much less regularization
        """Train MAP estimate for initialization"""
        
        if self.map_trained:
            return {}
        
        # Larger initialization
        self.U = torch.randn(self.n_users, self.k) * 0.5  # Much larger
        self.V = torch.randn(self.n_items, self.k) * 0.5
        
        self.U.requires_grad = True
        self.V.requires_grad = True
        
        # No weight decay in optimizer - we'll do it manually for better control
        optimizer = optim.Adam([self.U, self.V], lr=lr)
        
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Linear prediction (no sigmoid) for MAP
            pred = self.U @ self.V.T
            
            # Scale predictions to roughly match rating scale
            pred = torch.sigmoid(pred) * 9 + 1  # Map to [1, 10]
            
            # MSE loss only on observed entries
            diff = (pred - (R * 9 + 1)) * mask  # R is normalized, scale it back
            mse = (diff ** 2).sum() / mask.sum()
            
            # Very light regularization
            reg = lambda_reg * (self.U.norm() ** 2 + self.V.norm() ** 2) / (self.n_users + self.n_items)
            
            loss = mse + reg
            loss.backward()
            
            # No gradient clipping - let it learn
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % 10 == 0:
                u_var = self.U.var(dim=0).mean().item()
                v_var = self.V.var(dim=0).mean().item()
                print(f"MAP Epoch {epoch}: Loss = {loss.item():.4f}, MSE = {mse.item():.4f}, U_var = {u_var:.6f}, V_var = {v_var:.6f}")
        
        # Detach
        self.U = self.U.detach()
        self.V = self.V.detach()
        
        # Don't add noise if variance is reasonable
        final_u_var = self.U.var(dim=0).mean().item()
        final_v_var = self.V.var(dim=0).mean().item()
        
        if final_u_var < 0.01 or final_v_var < 0.01:
            print(f"WARNING: Low variance after MAP (U: {final_u_var:.6f}, V: {final_v_var:.6f})")
            print("Consider disabling MAP initialization or adjusting hyperparameters")
        
        self.map_trained = True
        
        if verbose:
            print(f"MAP training complete!")
            print(f"Final U variance: {final_u_var:.6f}")
            print(f"Final V variance: {final_v_var:.6f}")
        
        return {'losses': losses}
    
    def sample_user_factors(self, R: torch.Tensor, mask: torch.Tensor):
        """Sample user factors following Eq. 11-13 from the paper"""

        old_U = self.U.clone()
        
        for i in range(self.n_users):
            # Get observed items for user i
            observed_items = mask[i].nonzero(as_tuple=True)[0]
            
            if len(observed_items) == 0:
                # Sample from prior
                cov = torch.inverse(self.Lambda_U + torch.eye(self.k) * 1e-6)
                # Add regularization to ensure positive definiteness
                cov_reg = cov + torch.eye(self.k, device=cov.device) * 1e-8
                self.U[i] = dist.MultivariateNormal(self.mu_U, cov_reg).sample()
            else:
                # Compute posterior parameters (Eq. 12-13)
                V_obs = self.V[observed_items]  # (n_obs, k)
                r_obs = R[i, observed_items]  # (n_obs,)
                
                # Equation 12: Posterior precision
                Lambda_i = self.Lambda_U + self.alpha * (V_obs.T @ V_obs)
                
                # Equation 13: Posterior mean  
                b = self.Lambda_U @ self.mu_U + self.alpha * (V_obs.T @ r_obs)
                mu_i = torch.linalg.solve(Lambda_i + torch.eye(self.k) * 1e-6, b)
                
                # Sample from posterior
                try:
                    cov_i = torch.inverse(Lambda_i + torch.eye(self.k) * 1e-6)
                    # Add regularization to ensure positive definiteness
                    cov_i_reg = cov_i + torch.eye(self.k, device=cov_i.device) * 1e-6
                    self.U[i] = dist.MultivariateNormal(mu_i, cov_i_reg).sample()
                except:
                    # Fallback for numerical stability
                    self.U[i] = mu_i + torch.randn(self.k) * 0.01

        # Check if anything changed
        change = (self.U - old_U).norm()
        if change < 1e-6:
            print(f"WARNING: User factors barely changed! Change norm: {change}")
    
    def sample_item_factors(self, R: torch.Tensor, mask: torch.Tensor):
        """Sample item factors - symmetric to user factors"""
        
        for j in range(self.n_items):
            observed_users = mask[:, j].nonzero(as_tuple=True)[0]
            
            if len(observed_users) == 0:
                cov = torch.inverse(self.Lambda_V + torch.eye(self.k) * 1e-6)
                # Add regularization to ensure positive definiteness
                cov_reg = cov + torch.eye(self.k, device=cov.device) * 1e-8
                self.V[j] = dist.MultivariateNormal(self.mu_V, cov_reg).sample()
            else:
                U_obs = self.U[observed_users]
                r_obs = R[observed_users, j]
                
                Lambda_j = self.Lambda_V + self.alpha * (U_obs.T @ U_obs)
                b = self.Lambda_V @ self.mu_V + self.alpha * (U_obs.T @ r_obs)
                mu_j = torch.linalg.solve(Lambda_j + torch.eye(self.k) * 1e-6, b)
                
                try:
                    cov_j = torch.inverse(Lambda_j + torch.eye(self.k) * 1e-6)
                    # Add regularization to ensure positive definiteness
                    cov_j_reg = cov_j + torch.eye(self.k, device=cov_j.device) * 1e-6
                    self.V[j] = dist.MultivariateNormal(mu_j, cov_j_reg).sample()
                except:
                    self.V[j] = mu_j + torch.randn(self.k) * 0.01
    
    def sample_user_hyperparams(self):
        """Sample hyperparameters following Eq. 14 from the paper"""
        N = self.n_users
        
        # Compute sufficient statistics
        U_bar = self.U.mean(0)
        S = (self.U.T @ self.U) / N
        
        # Posterior parameters for Normal-Wishart
        beta_star = self.beta_0 + N
        mu_star = (self.beta_0 * self.mu_0 + N * U_bar) / beta_star
        nu_star = self.nu_0 + N
        
        # Update W_star (Eq. 14)
        diff = U_bar - self.mu_0
        W_star_inv = torch.inverse(self.W_0) + N * S + \
                     (self.beta_0 * N / beta_star) * torch.outer(diff, diff)
        
        # Ensure numerical stability
        W_star_inv = (W_star_inv + W_star_inv.T) / 2
        W_star = torch.inverse(W_star_inv + torch.eye(self.k) * 1e-6)
        W_star = (W_star + W_star.T) / 2  # Ensure symmetry
        
        # Sample Lambda_U from Wishart
        try:
            # Ensure positive definite
            eigvals = torch.linalg.eigvalsh(W_star)
            if eigvals.min() > 1e-6:
                self.Lambda_U = torch.from_numpy(
                    stats.wishart.rvs(df=nu_star, scale=W_star.numpy())
                ).float()
            else:
                self.Lambda_U = torch.eye(self.k) * 2.0
        except:
            self.Lambda_U = torch.eye(self.k) * 2.0
        
        # Sample mu_U from Normal
        try:
            cov_mu = torch.inverse(beta_star * self.Lambda_U + torch.eye(self.k) * 1e-6)
            # Add regularization to ensure positive definiteness
            cov_mu_reg = cov_mu + torch.eye(self.k, device=cov_mu.device) * 1e-6
            self.mu_U = dist.MultivariateNormal(mu_star, cov_mu_reg).sample()
        except:
            self.mu_U = mu_star
    
    def sample_item_hyperparams(self):
        """Sample item hyperparameters - symmetric to user hyperparams"""
        M = self.n_items
        
        V_bar = self.V.mean(0)
        S = (self.V.T @ self.V) / M
        
        beta_star = self.beta_0 + M
        mu_star = (self.beta_0 * self.mu_0 + M * V_bar) / beta_star
        nu_star = self.nu_0 + M
        
        diff = V_bar - self.mu_0
        W_star_inv = torch.inverse(self.W_0) + M * S + \
                     (self.beta_0 * M / beta_star) * torch.outer(diff, diff)
        
        W_star_inv = (W_star_inv + W_star_inv.T) / 2
        W_star = torch.inverse(W_star_inv + torch.eye(self.k) * 1e-6)
        W_star = (W_star + W_star.T) / 2
        
        try:
            eigvals = torch.linalg.eigvalsh(W_star)
            if eigvals.min() > 1e-6:
                self.Lambda_V = torch.from_numpy(
                    stats.wishart.rvs(df=nu_star, scale=W_star.numpy())
                ).float()
            else:
                self.Lambda_V = torch.eye(self.k) * 2.0
        except:
            self.Lambda_V = torch.eye(self.k) * 2.0
        
        try:
            cov_mu = torch.inverse(beta_star * self.Lambda_V + torch.eye(self.k) * 1e-6)
            # Add regularization to ensure positive definiteness
            cov_mu_reg = cov_mu + torch.eye(self.k, device=cov_mu.device) * 1e-6
            self.mu_V = dist.MultivariateNormal(mu_star, cov_mu_reg).sample()
        except:
            self.mu_V = mu_star
    
    def gibbs_sample(self, R: torch.Tensor, mask: torch.Tensor,
                     R_test: Optional[torch.Tensor] = None,
                     mask_test: Optional[torch.Tensor] = None,
                     n_samples: int = 1500, 
                     burn_in: int = 500,
                     use_map_init: bool = True,
                     check_convergence: bool = True,
                     min_samples: int = 200,
                     check_every: int = 10,
                     verbose: bool = True) -> Dict:
        """
        Adaptive Gibbs sampling with optional convergence monitoring
        
        Args:
            R: Training ratings matrix (normalized to [0,1])
            mask: Training mask indicating observed ratings
            R_test: Test ratings for convergence monitoring
            mask_test: Test mask
            n_samples: Maximum number of samples to collect
            burn_in: Number of burn-in samples
            use_map_init: Whether to use MAP initialization
            check_convergence: Whether to use adaptive stopping
            min_samples: Minimum samples before checking convergence
            check_every: How often to check convergence
            verbose: Whether to print progress
        """
        print(f"Input R device: {R.device}")
        print(f"Input mask device: {mask.device}")
        print(f"Model U device: {self.U.device}")
        print(f"Model V device: {self.V.device}")

        # MAP initialization if requested
        if use_map_init and not self.map_trained:
            print("Training MAP initialization...")
            self.train_map(R, mask, verbose=verbose)
        
        # Initialize convergence monitor if needed
        monitor = ConvergenceMonitor() if check_convergence else None
        
        # Storage for samples
        samples = {
            'U': [], 'V': [], 
            'mu_U': [], 'mu_V': [], 
            'Lambda_U': [], 'Lambda_V': []
        }
        
        # Progress bar
        total_iterations = n_samples + burn_in
        pbar = tqdm(total=total_iterations, desc="Gibbs Sampling")
        
        converged = False
        actual_samples = 0
        
        for t in range(total_iterations):
            # Gibbs sampling steps (following paper's algorithm)
            self.sample_user_hyperparams()
            self.sample_item_hyperparams()
            self.sample_user_factors(R, mask)
            self.sample_item_factors(R, mask)
            
            # Store samples after burn-in
            if t >= burn_in:
                samples['U'].append(self.U.clone())
                samples['V'].append(self.V.clone())
                samples['mu_U'].append(self.mu_U.clone())
                samples['mu_V'].append(self.mu_V.clone())
                samples['Lambda_U'].append(self.Lambda_U.clone())
                samples['Lambda_V'].append(self.Lambda_V.clone())
                actual_samples += 1
                
                # Check convergence if enabled
                if (check_convergence and 
                    actual_samples >= min_samples and 
                    actual_samples % check_every == 0):
                    
                    converged, diagnostics = monitor.check_convergence(
                        self.U, self.V, R, mask, R_test, mask_test
                    )
                    
                    # Update progress bar
                    postfix = {
                        'samples': actual_samples,
                        'LL': f"{diagnostics['log_likelihood']:.1f}",
                    }
                    if diagnostics['rmse_test'] is not None:
                        postfix['RMSE'] = f"{diagnostics['rmse_test']:.4f}"
                    if 'geweke_z' in diagnostics:
                        postfix['Geweke'] = f"{diagnostics['geweke_z']:.2f}"
                    
                    pbar.set_postfix(postfix)
                    
                    if converged:
                        print(f"\nConverged after {t+1} iterations ({actual_samples} samples)!")
                        if verbose:
                            for key, value in diagnostics.items():
                                if value is not None:
                                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                        break
            
            pbar.update(1)
        
        pbar.close()
        
        if not converged and check_convergence:
            print(f"Reached maximum iterations without convergence")
        
        result = {
            'samples': samples,
            'n_iterations': t + 1,
            'n_samples': actual_samples,
            'converged': converged
        }
        
        if monitor is not None:
            result['diagnostics'] = dict(monitor.metrics_history)
        
        return result
