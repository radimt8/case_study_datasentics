# Bayesian Matrix Factorization for Book Recommendations

## Executive Summary

This project implements a **Bayesian Matrix Factorization system** for book recommendations using **MAP-initialized MCMC with Gibbs sampling**, developed as a case study demonstrating advanced machine learning techniques with microservices architecture. The implementation incorporates uncertainty quantification, convergence monitoring, hyperparameter optimization, and cold-start user support.

## System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Optimizer     │    │   Batch Service  │    │   API Service   │
│   (Optuna)      │───▶│   (MCMC Train)   │    │   (FastAPI)     │
│                 │    │                  │    │                 │
│ • Hyperparams   │    │ • Gibbs Sampling │    │ • Recommendations│
│ • Bayesian Opt  │    │ • Convergence    │◄───┤ • Cold-start    │
│ • Auto Config   │    │ • Precompute     │    │ • Uncertainty   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                  │                       ▲
                                  ▼                       │
                           ┌──────────────────┐           │
                           │   DragonflyDB    │───────────┘
                           │   (Storage)      │    
                           │                  │    ┌─────────────────┐
                           │ • Model State    │───▶│   Frontend      │
                           │ • Precomputed    │    │   (Web UI)      │
                           │   Recommendations│    │                 │
                           └──────────────────┘    │ • User Interface│
                                                   │ • Book Search   │
                                                   │ • Interactions  │
                                                   └─────────────────┘
```

**For detailed setup and deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

## Technical Implementation

### Dataset and Preprocessing
- **Source:** BookCrossing dataset (278K users, 271K books, 1.15M ratings)
- **Filtering Strategy:** Active users (≥5 ratings) and popular books (≥10 ratings)
- **Final Dataset:** 11,928 users × 5,710 books with 120,035 ratings (~99.8% sparse)
- **Split:** 80/20 train/test for generalization assessment

### Mathematical Framework

**Bayesian Matrix Factorization Model:**
```
R_ij ~ N(σ(U_i^T V_j), σ_noise²)
U_i ~ N(μ_U, Λ_U^-1)
V_j ~ N(μ_V, Λ_V^-1)
```

**MCMC Implementation (Gibbs Sampling):**
- MAP initialization for better convergence
- Adaptive sampling with convergence monitoring
- Hyperparameter sampling using Normal-Wishart priors
- Geweke diagnostics and parameter stability tracking
- Early stopping based on test RMSE patience

**Hyperparameter Optimization:**
- Optuna-based Bayesian optimization
- Optimizes: latent factors (k), observation precision (α), sample counts, burn-in
- Multi-objective: user diversity + prediction accuracy
- Automatically updates environment configuration

### Performance Metrics

**Current Optimized Configuration:**
- Latent factors (k): 5
- Observation precision (α): 6.57
- MCMC samples: 550
- Burn-in: 200
- Adaptive convergence monitoring enabled

**Note:** Performance metrics will vary based on hyperparameter optimization results. The system uses Optuna to automatically find optimal configurations balancing user diversity and prediction accuracy.

## System Architecture

### Microservices Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Batch Service │    │   API Service    │    │   Frontend      │
│   (Training)    │───▶│   (FastAPI)      │───▶│   (Web UI)      │
│                 │    │                  │    │                 │
│ • MCMC Training │    │ • Recommendations│    │ • User Interface│
│ • Optimization  │    │ • Cold-start     │    │ • Book Search   │
│ • Precomputation│    │ • Uncertainty    │    │ • Real-time API │
└─────────────────┘    └──────────────────┘    └─────────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                           ┌──────────────────┐
                           │   DragonflyDB    │
                           │   (Redis-compat) │
                           │                  │
                           │ • Model Storage  │
                           │ • Precomputed    │
                           │   Recommendations│
                           └──────────────────┘
```

### Implemented Features

#### Core ML Pipeline
1. **MCMC Training:** MAP-initialized Gibbs sampling with convergence monitoring
2. **Cold-Start Support:** Weighted factor imputation for filtered users  
3. **Uncertainty Quantification:** Credible intervals for all predictions
4. **Hyperparameter Optimization:** Automated tuning with Optuna

#### System Services
5. **Real-time API:** FastAPI with comprehensive error handling
6. **Web Interface:** Complete frontend for user interaction
7. **Scalable Storage:** DragonflyDB for model and recommendation storage
8. **Docker Deployment:** Full containerized architecture

## Critical Assessment

### Key Strengths
1. **Mathematical Framework:** Full Bayesian inference with uncertainty quantification
2. **Convergence Monitoring:** Geweke diagnostics and parameter stability tracking
3. **Cold-Start Handling:** Expanded coverage from ~5% to 100% of user base
4. **Automated Optimization:** Optuna-based hyperparameter search
5. **Complete Pipeline:** End-to-end system from training to user interface

### Limitations and Trade-offs

#### 1. **Convergence Challenges**
- **Reality:** MCMC chains likely not fully converged with current sample counts (550 samples)
- **Evidence:** True convergence typically requires thousands to tens of thousands of samples
- **Trade-off:** Current approach provides reasonable results for PoC, but not mathematically rigorous
- **Impact:** Uncertainty estimates may be underestimated

#### 2. **Data Sparsity** 
- **Issue:** 99.8% sparsity remains fundamental limitation
- **Mitigation:** Relaxed filtering increased dataset 4x, but sparsity persists
- **Reality:** Collaborative filtering inherently limited by observation density

#### 3. **Cold-Start Quality**
- **Implementation:** Weighted factor imputation provides coverage but lower quality
- **Limitation:** No content features or sophisticated cold-start modeling
- **Trade-off:** Coverage vs. recommendation quality

#### 4. **Computational Cost**
- **Training:** MCMC sampling requires significant compute time
- **Inference:** Monte Carlo uncertainty estimation adds latency
- **Storage:** Precomputed recommendations scale with user base

## Technical Considerations

### Current State
- **Proof of Concept:** Demonstrates full ML pipeline with uncertainty quantification
- **Architecture:** Complete microservices setup with API, frontend, and storage
- **Coverage:** Handles both trained users and cold-start scenarios
- **Automation:** Hyperparameter optimization and containerized deployment

### Areas for Enhancement
1. **Convergence:** Longer MCMC chains for mathematical rigor
2. **Content Features:** Incorporate book metadata and user demographics  
3. **Scalability:** Approximate methods for faster uncertainty estimation
4. **Evaluation:** A/B testing framework and business metric tracking

## Known Limitations and Model Behavior

### Collaborative Filtering Challenges
This implementation demonstrates classic collaborative filtering limitations in practice:

**Lack of Semantic Understanding:** Without content-based features, the model cannot capture semantic similarity. For example, a user rating "The Return of the King" 10/10 may not receive recommendations for "The Fellowship of the Ring" or "The Two Towers" if other users' rating patterns don't align perfectly.

**Factor Collapse Issues:** Analysis of the learned factors reveals potential convergence to degenerate solutions where:
- Cosine similarities between different book types approach 1.0
- The model struggles to distinguish between fantasy novels, self-help books, and children's literature  
- User and item factors operate at mismatched scales (observed U/V norm ratios of 60:1+)

**Local Minima Convergence:** The optimization may get trapped in poor local minima where:
- All books appear essentially equivalent to the model
- High uncertainty values (±1.4-1.5 rating points) indicate low model confidence
- Factor diversity is insufficient with k=5 latent dimensions

### Mathematical Framework vs. Practical Results
While the Bayesian framework is mathematically sound, the learned representations can collapse to a single mode. This is a well-documented failure mode in matrix factorization when optimization converges to suboptimal solutions.

**Current Parameter Impact (α=6.57):**
- Higher α values from optimization may help distinguish rating differences
- However, insufficient factor diversity (k=5) limits the model's representational capacity
- MAP initialization, while theoretically beneficial, may inadvertently guide toward degenerate solutions

### Potential Solutions Not Implemented
1. **Increased Dimensionality:** k=20-50 factors to capture book type diversity
2. **Content Integration:** Book metadata (genre, author, series) for semantic similarity
3. **Hybrid Approaches:** Combine collaborative and content-based methods
4. **Advanced Initialization:** Alternative to MAP that encourages factor diversity
5. **Regularization Tuning:** Prevent factor collapse through targeted constraints

This case study demonstrates both the theoretical elegance and practical challenges of pure collaborative filtering approaches in sparse, diverse domains.

## Business Value Proposition

1. **Risk-Aware Recommendations:** Uncertainty quantification enables business decisions about recommendation confidence
2. **Model Transparency:** Diagnostic framework provides interpretability for stakeholder trust
3. **Systematic Evaluation:** Rigorous methodology ensures reliable performance claims
4. **Production Scalability:** Clear path from prototype to production system

## Repository Structure

```
.
├── data/                   # BookCrossing dataset
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
├── notebooks/
│   └── Untitled.ipynb    # Main implementation notebook
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Service orchestration
└── README.md             # This documentation
```

## Getting Started

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   # or
   docker-compose up
   ```

2. **Run Analysis:**
   ```bash
   jupyter notebook notebooks/Untitled.ipynb
   ```

3. **Key Functions:**
   - `train_bayesian_mf()`: Main training loop
   - `make_recommendations_with_uncertainty()`: Prediction with uncertainty
   - `quantify_sigmoid_bias()`: Approximation quality assessment

## Conclusion

This implementation demonstrates how mathematical rigor can provide business value through uncertainty quantification and systematic evaluation. While fundamental limitations (sparsity, cold-start) constrain pure collaborative filtering approaches, the diagnostic framework and production-ready architecture provide a solid foundation for hybrid recommendation systems.

**The key insight:** Advanced mathematical techniques are most valuable when they illuminate fundamental limitations rather than obscuring them, enabling informed business decisions about model deployment and improvement priorities.

---

*Developed as a case study demonstrating advanced machine learning techniques with production considerations for data consulting applications.*
