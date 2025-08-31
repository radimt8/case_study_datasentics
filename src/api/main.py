from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import os
from typing import List, Optional
from pydantic import BaseModel
import pickle
from fuzzywuzzy import fuzz, process

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

class UserRating(BaseModel):
    title: str
    rating: float

class ColdStartRequest(BaseModel):
    ratings: List[UserRating]
    top_k: Optional[int] = 10

class BookSearchResponse(BaseModel):
    title: str
    similarity: float

class BookSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

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
    """Get list of users with precomputed recommendations (includes both trained and cold start users)"""
    try:
        # Get all recommendation keys from Redis
        keys = redis_client.keys("recs:*")
        user_ids = [int(key.split(":")[1]) for key in keys]
        user_ids.sort()
        
        return {
            "total_users": len(user_ids),
            "user_ids": user_ids[:100],  # Return first 100 for API responsiveness
            "note": "Includes both trained users and cold start users"
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

def _load_recommender():
    """Load the trained recommender model from Redis"""
    try:
        # Check if model data exists
        model_data = redis_client.get('model:recommender')
        if not model_data:
            return None
        
        # Deserialize the recommender
        recommender = pickle.loads(model_data.encode('latin1'))
        return recommender
    except Exception as e:
        print(f"Error loading recommender: {e}")
        return None

def _get_book_titles():
    """Get list of all book titles from Redis"""
    try:
        book_titles_data = redis_client.get('model:book_titles')
        if not book_titles_data:
            return []
        return json.loads(book_titles_data)
    except Exception:
        return []

@app.post("/recommend/cold-start")
async def cold_start_recommendations(request: ColdStartRequest):
    """Get recommendations for new user based on their ratings"""
    
    # Load the recommender
    recommender = _load_recommender()
    if not recommender:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please run batch processing first."
        )
    
    try:
        # Convert request to format expected by recommender
        user_ratings = [{"title": r.title, "rating": r.rating} for r in request.ratings]
        
        # Get recommendations
        result = recommender.cold_start_recommendations(user_ratings, request.top_k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "user_type": "cold_start",
            "recommendations": result["recommendations"],
            "total_available": len(result["recommendations"]),
            "valid_books_rated": result.get("valid_books_rated", 0),
            "note": f"Based on {result.get('valid_books_rated', 0)} books found in our dataset"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/books/search")
async def search_books(request: BookSearchRequest):
    """Search for books using fuzzy matching"""
    
    book_titles = _get_book_titles()
    if not book_titles:
        raise HTTPException(
            status_code=503,
            detail="Book data not available. Please run batch processing first."
        )
    
    try:
        # Use fuzzywuzzy to find similar book titles
        matches = process.extract(request.query, book_titles, limit=request.limit)
        
        # Convert to response format
        results = [
            BookSearchResponse(title=match[0], similarity=match[1]/100.0)
            for match in matches
        ]
        
        return {
            "query": request.query,
            "results": results,
            "total_books": len(book_titles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching books: {str(e)}")

@app.get("/books")
async def get_all_books(limit: Optional[int] = 100, offset: Optional[int] = 0):
    """Get paginated list of all available books"""
    
    book_titles = _get_book_titles()
    if not book_titles:
        raise HTTPException(
            status_code=503,
            detail="Book data not available. Please run batch processing first."
        )
    
    # Paginate results
    start_idx = offset
    end_idx = offset + limit
    paginated_books = book_titles[start_idx:end_idx]
    
    return {
        "books": paginated_books,
        "total_books": len(book_titles),
        "offset": offset,
        "limit": limit,
        "has_more": end_idx < len(book_titles)
    }