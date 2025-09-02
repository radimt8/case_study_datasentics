# Deployment Guide

## Performance Considerations

**CUDA vs CPU:** After testing both CUDA and CPU implementations, we found that **CPU training is actually faster** for this workload size. While the base Docker images include CUDA support, GPU acceleration proved counterproductive due to:
- Small matrix dimensions (11K users × 5K books) 
- MCMC sampling overhead outweighing GPU benefits
- Memory transfer costs between CPU/GPU for Gibbs sampling

**Image Optimization:** The system uses multi-stage Docker builds to avoid downloading PyTorch and heavy dependencies multiple times across services, saving significant bandwidth and build time.

## Microservices Architecture

This system consists of five main services:

1. **Optimizer Service**: Optuna-based hyperparameter optimization
2. **Batch Service**: MCMC training and recommendation precomputation
3. **API Service**: FastAPI server that serves recommendations  
4. **Frontend Service**: Web interface for user interactions
5. **DragonflyDB**: High-performance Redis-compatible data store

## Quick Start

### 1. Build Base Image (Required First Step)
```bash
# MANDATORY: Build the base image with all heavy dependencies first
docker build -f Dockerfile.base -t bayesian-mcmc:base .
```

**CUDA Compatibility Note:** Due to nvidia-container-toolkit compatibility issues with Linux 6.16+:
- For CUDA acceleration: Use `podman` instead of `docker`, or set up `micromamba` environment on host
- **Recommended:** Use CPU-only by changing base image to standard Python (CUDA is slower anyway for this workload)

### 1b. Start Core Services
```bash
# Start runtime services
docker-compose up -d  # (or explicitly: docker-compose up -d api frontend dragonfly)

# Check services are running
docker-compose ps
```

**Note:** The `batch` and `optimizer` services have profiles defined in docker-compose.yml, so they won't start automatically. They must be run explicitly using `docker-compose run --rm <service>` commands.

### 2. Hyperparameter Optimization (Optional)
```bash
# Find optimal hyperparameters using Optuna
docker-compose run --rm optimizer

# Or set custom number of optimization trials
OPTIMIZER_N_CALLS=20 docker-compose run --rm optimizer
```

### 3. Start All Services
```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps
```

### 4. Run MCMC Training
```bash
# Run batch processing to train model and generate recommendations
docker-compose run --rm batch
```

### 3. Test API
```bash
# Check API health
curl http://localhost:8000/health

# Get model information
curl http://localhost:8000/model/info

# Get recommendations for a user
curl http://localhost:8000/recommend/8

# List available users
curl http://localhost:8000/users
```

## Optimized Docker Build

To avoid downloading PyTorch (~2GB+) multiple times across different containers, use the multi-stage base image approach:

### Build Base Image First
```bash
# Build the base image with all heavy dependencies (PyTorch, NumPy, etc.)
docker build -f Dockerfile.base -t bayesian-mcmc:base .
```

### Then Build Service Images
```bash
# Now all service builds are lightning fast (only copy source code)
docker-compose build

# Or build individual services
docker build -f Dockerfile.api -t bayesian-mcmc-api .
docker build -f Dockerfile.batch -t bayesian-mcmc-batch .
docker build -f Dockerfile.optimizer -t bayesian-mcmc-optimizer .
```

### Benefits
- **Bandwidth Savings**: Download PyTorch once, reuse across all services
- **Faster Iteration**: Service rebuilds only copy source code (seconds vs minutes)
- **Build Consistency**: Same dependency versions across all containers
- **GPU Ready**: Especially valuable when preparing CUDA-enabled builds if GPU is available

### Development Workflow
1. Build base image once when setting up or updating dependencies
2. Use `docker-compose build` for fast service updates during development
3. Only rebuild base image when requirements change

## Service Details

### API Service (Port 8000)

**Endpoints:**
- `GET /` - API welcome message
- `GET /health` - Health check with Redis connectivity
- `GET /recommend/{user_id}?top_k=10` - Get recommendations for user
- `GET /users` - List users with available recommendations
- `GET /stats` - System statistics
- `GET /model/info` - Model metadata and performance metrics

**Example Response:**
```json
{
  "user_id": 8,
  "recommendations": [
    {
      "title": "The Lovely Bones: A Novel",
      "predicted_rating": 7.2,
      "uncertainty": 1.1,
      "confidence_interval": [5.04, 9.36]
    }
  ],
  "total_available": 10
}
```

### Batch Service

**What it does:**
- Loads BookCrossing dataset (278K users, 271K books, 1.15M ratings)
- Filters to active users (≥20 ratings) and popular books (≥50 ratings)  
- Trains Bayesian Matrix Factorization with variational inference
- Generates uncertainty-quantified recommendations for all users
- Stores precomputed results in Dragonfly

**Performance:**
- Training: ~2-10 minutes depending on hardware
- Recommendation generation: ~5-15 minutes for all users
- Storage: ~5-10MB total in Dragonfly

### Dragonfly Service (Port 6379)

- High-performance Redis-compatible data store
- 25x faster than Redis for concurrent reads
- Perfect for serving precomputed recommendations at scale

## Development Workflow

### Development Mode (with Jupyter)
```bash
# Start all services including Jupyter
docker-compose up -d

# Access Jupyter at http://localhost:8888
# Work on notebooks in ./notebooks/
```

### Production Mode
```bash
# API + Dragonfly only
docker-compose up -d api dragonfly

# Run batch processing via cron or scheduler
# Example: Every 6 hours
0 */6 * * * docker-compose run --rm batch
```

### Monitoring
```bash
# View logs
docker-compose logs api
docker-compose logs batch

# Monitor Dragonfly
docker-compose exec dragonfly dragonfly --logtostderr --alsologtostderr --v=1
```

## Scaling Considerations

### High Availability
- Run multiple API instances behind load balancer
- Use Dragonfly clustering for data redundancy
- Separate batch processing to dedicated machines

### Performance Optimization
- Batch processing: Run during off-peak hours
- API serving: Consider CDN for static responses
- Monitoring: Set up alerts for batch job failures

### Resource Requirements

**Minimum:**
- API: 512MB RAM, 1 CPU
- Batch: 2GB RAM, 2 CPU (during training)
- Dragonfly: 256MB RAM, 1 CPU

**Recommended:**
- API: 1GB RAM, 2 CPU
- Batch: 4GB RAM, 4 CPU
- Dragonfly: 512MB RAM, 2 CPU

## Production Limitations

**⚠️ Important:** This system currently serves only ~4% of the original user base (users with ≥20 ratings). See [README.md - Cold-Start Coverage Gap](README.md#major-limitation-cold-start-user-coverage-gap) for detailed analysis and solutions.

**Impact on deployment:**
- **User Coverage:** Can only serve recommendations to ~3K out of 278K total users
- **API Responses:** Most user IDs will return 404 "No recommendations found" 
- **Business Use:** Suitable for demonstrating technical capabilities, not production traffic

**Next Steps:** Implement weighted factor imputation and hybrid content-based recommendations to achieve 100% user coverage.

## Troubleshooting

### Common Issues

**1. "Model not trained yet" error**
```bash
# Solution: Run batch processing first
docker-compose run --rm batch
```

**2. Redis connection failed**
```bash
# Check Dragonfly is running
docker-compose ps dragonfly

# Restart if needed
docker-compose restart dragonfly
```

**3. Out of memory during batch processing**
```bash
# Reduce sample size in batch processor
# Edit src/batch/processor.py: n_samples=100 (default: 500)
```

**4. No recommendations for user**
- User may not exist in filtered dataset (needs ≥20 ratings)
- Check `/users` endpoint for valid user IDs

### Logs and Debugging
```bash
# API logs
docker-compose logs -f api

# Batch processing logs  
docker-compose logs batch

# All services
docker-compose logs -f
```