from fastapi import FastAPI, Response, HTTPException, Depends, Query

# Cross-Origin Resource Sharing
#   allows frontend to talk to backend
#   without this, browser blocks requests
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone

from config.settings import get_settings
from backend.db.database import get_db
from backend.db.models import Article, User, Interaction

from backend.api.schemas import (
    ArticleResponse,
    ArticleDetailResponse,
    InteractionCreate,
    InteractionResponse,
    SuccessResponse,
    HealthResponse,
    CountResponse,
)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="News Recommender API",
    version="0.1.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "message": "News Recommender API is running",
        "timestamp": datetime.now(timezone.utc)
    }

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc)
    }

'''
# @app.get("/health")
# async def health_check(db: Session = Depends(get_db)):
#     try:
#         # Check database connection
#         db.execute("SELECT 1")
        
#         # Check Redis connection
#         redis_client.ping()
        
#         # Check model loaded
#         if not classifier_model:
#             raise Exception("Model not loaded")
        
#         return {
#             "status": "ok",
#             "database": "connected",
#             "redis": "connected",
#             "model": "loaded"
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }
'''

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)



# ===========================================================
# ==================== Article Endpoints ====================
# ===========================================================

@app.get(
    "/api/articles",
    response_model=List[ArticleResponse],
    tags=["articles"],
    summary="Get articles",
    description="Retrieve a paginated list of articles with optional filters"
)
async def get_articles(
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of articles to return"),
    topic: str | None = Query(None, description="Filter by topic (world, sports, business, tech)"),
    source: str | None = Query(None, description="Filter by source URL"),
    db: Session = Depends(get_db)
):
    """
    Get a list of articles with pagination and optional filters.
    
    - **skip**: Number of articles to skip (for pagination)
    - **limit**: Maximum articles to return (1-100)
    - **topic**: Filter by topic category
    - **source**: Filter by source URL (partial match)
    """
    query = db.query(Article)
    
    # Apply filters
    if topic:
        query = query.filter(Article.topic == topic)
    if source:
        query = query.filter(Article.source.contains(source))
    
    # Order by most recent first
    query = query.order_by(Article.scraped_at.desc())
    
    # Pagination
    articles = query.offset(skip).limit(limit).all()
    
    return articles


@app.get(
    "/api/articles/count",
    response_model=CountResponse,
    tags=["articles"],
    summary="Get article count",
    description="Get total number of articles in database"
)
async def get_article_count(
    topic: str | None = Query(None, description="Filter count by topic"),
    db: Session = Depends(get_db)
):
    """
    Get total count of articles, optionally filtered by topic.
    """
    query = db.query(Article)
    
    if topic:
        query = query.filter(Article.topic == topic)
    
    count = query.count()
    return {"count": count}


@app.get(
    "/api/articles/recent",
    response_model=List[ArticleResponse],
    tags=["articles"],
    summary="Get recent articles",
    description="Get the most recently scraped articles"
)
async def get_recent_articles(
    limit: int = Query(10, ge=1, le=50, description="Number of recent articles"),
    db: Session = Depends(get_db)
):
    """
    Get the most recently scraped articles.
    
    - **limit**: Number of articles to return (1-50)
    """
    articles = db.query(Article)\
        .order_by(Article.scraped_at.desc())\
        .limit(limit)\
        .all()
    
    return articles


@app.get(
    "/api/articles/{article_id}",
    response_model=ArticleDetailResponse,
    tags=["articles"],
    summary="Get article by ID",
    description="Retrieve full details of a single article"
)
async def get_article(
    article_id: int,
    db: Session = Depends(get_db)
):
    """
    Get full details of a specific article by ID.
    
    - **article_id**: The ID of the article to retrieve
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    
    if not article:
        raise HTTPException(
            status_code=404,
            detail=f"Article with ID {article_id} not found"
        )
    
    return article

# ===============================================================
# ==================== Interaction Endpoints ====================
# ===============================================================

@app.post(
    "/api/interaction",
    response_model=SuccessResponse,
    tags=["interactions"],
    summary="Log user interaction",
    description="Record a user interaction with an article"
)
async def log_interaction(
    interaction: InteractionCreate,
    db: Session = Depends(get_db)
):
    """
    Log a user interaction with an article.
    
    Interaction types:
    - **click**: User clicked on article
    - **read**: User read the article
    - **like**: User liked the article
    - **skip**: User skipped the article
    """
    # Create database object
    db_interaction = Interaction(**interaction.model_dump())
    
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    
    return {
        "status": "success",
        "message": f"Interaction logged: {interaction.interaction_type}"
    }


@app.get(
    "/api/interactions/{user_id}",
    response_model=List[InteractionResponse],
    tags=["interactions"],
    summary="Get user interactions",
    description="Retrieve all interactions for a specific user"
)
async def get_user_interactions(
    user_id: int,
    limit: int = Query(50, ge=1, le=200, description="Maximum interactions to return"),
    db: Session = Depends(get_db)
):
    """
    Get interaction history for a specific user.
    
    - **user_id**: The ID of the user
    - **limit**: Maximum number of interactions to return
    """
    interactions = db.query(Interaction)\
        .filter(Interaction.user_id == user_id)\
        .order_by(Interaction.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return interactions


# ========================================================
# ==================== Stats Endpoint ====================
# ========================================================

@app.get(
    "/api/stats",
    tags=["stats"],
    summary="Get system statistics",
    description="Get overall system statistics"
)
async def get_stats(db: Session = Depends(get_db)):
    """
    Get system-wide statistics.
    """
    article_count = db.query(Article).count()
    user_count = db.query(User).count()
    interaction_count = db.query(Interaction).count()
    
    # Get topic distribution
    topics = db.query(Article.topic, db.func.count(Article.id))\
        .filter(Article.topic.isnot(None))\
        .group_by(Article.topic)\
        .all()
    
    topic_distribution = {topic: count for topic, count in topics}
    
    return {
        "total_articles": article_count,
        "total_users": user_count,
        "total_interactions": interaction_count,
        "topic_distribution": topic_distribution,
        "timestamp": datetime.now(timezone.utc)
    }





if __name__ == "__main__":
    import uvicorn

    '''
    "backend.api.main:app"
        ↓         ↓     ↓
    package   module  variable

    Means:
    - Go to backend/api/main.py
    - Find the variable named "app"
    - That's the FastAPI instance to run
    '''
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )