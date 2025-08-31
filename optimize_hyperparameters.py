#!/usr/bin/env python3
"""
Hyperparameter Optimization Script

Usage:
    python optimize_hyperparameters.py

Environment Variables:
    OPTIMIZER_N_CALLS: Number of optimization iterations (default: 15)
"""

if __name__ == "__main__":
    from src.optimizer.hyperparameter_optimizer import main
    main()