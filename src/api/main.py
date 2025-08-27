from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import os
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(
    title="Bayesian Book Recommendations",
    description="Book recommendation system with uncertainty quantification",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url, decode_responses=True)

# Response models
class Recommendation(BaseModel):
    title: str
    predicted_rating: float
    uncertainty: float
    confidence_interval: List[float]

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]
    total_available: int
    note: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Bayesian Book Recommendation API"}

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": "disconnected", "error": str(e)}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_k: int = 10):
    """Get book recommendations for a user with uncertainty quantification"""
    
    # Fetch precomputed recommendations from Dragonfly
    data = redis_client.get(f"recs:{user_id}")
    
    if not data:
        raise HTTPException(
            status_code=404, 
            detail=f"No recommendations found for user {user_id}"
        )
    
    try:
        all_recommendations = json.loads(data)
        max_available = len(all_recommendations)
        
        # Adjust top_k if requesting more than available
        actual_k = min(top_k, max_available)
        
        # Return available recommendations
        recommendations = [
            Recommendation(**rec) for rec in all_recommendations[:actual_k]
        ]
        
        response = RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            total_available=max_available
        )
        
        # Add note if user requested more than available
        if top_k > max_available:
            response.note = f"Requested {top_k} recommendations, but only {max_available} are precomputed. To get more, retrain the model with higher top_k_recs parameter."
        
        return response
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid recommendation data format"
        )

@app.get("/users")
async def get_available_users():
    """Get list of users with precomputed recommendations"""
    try:
        # Get all recommendation keys from Redis
        keys = redis_client.keys("recs:*")
        user_ids = [int(key.split(":")[1]) for key in keys]
        user_ids.sort()
        
        return {
            "total_users": len(user_ids),
            "user_ids": user_ids[:100]  # Return first 100 for API responsiveness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        total_keys = len(redis_client.keys("recs:*"))
        memory_usage = redis_client.info("memory")
        
        return {
            "total_users_with_recs": total_keys,
            "memory_usage_mb": round(memory_usage["used_memory"] / 1024 / 1024, 2),
            "redis_version": redis_client.info("server")["redis_version"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get model metadata and performance metrics"""
    try:
        metadata = redis_client.get("model:metadata")
        if not metadata:
            raise HTTPException(
                status_code=404, 
                detail="Model not trained yet. Run batch processing first."
            )
        
        return json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid model metadata")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))