from fastapi import FastAPI, Response, HTTPException, Depends, Query

# Cross-Origin Resource Sharing
#   allows frontend to talk to backend
#   without this, browser blocks requests
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
from sqlalchemy import func
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
    ClassifyTextRequest,
    ClassifyArticleRequest,
    ClassificationResult,
    BatchClassifyRequest,
    BatchClassificationResponse,
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
    # Validate that user exists
    user = db.query(User).filter(User.id == interaction.user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {interaction.user_id} not found"
        )
    
    # Validate that article exists
    article = db.query(Article).filter(Article.id == interaction.article_id).first()
    if not article:
        raise HTTPException(
            status_code=404,
            detail=f"Article with ID {interaction.article_id} not found"
        )
    
    # Handle different interaction types with appropriate logic
    if interaction.interaction_type == "like":
        # For likes, check if already liked (prevent duplicate likes)
        existing_like = db.query(Interaction).filter(
            Interaction.user_id == interaction.user_id,
            Interaction.article_id == interaction.article_id,
            Interaction.interaction_type == "like"
        ).first()
        
        if existing_like:
            # User already liked - remove the like (toggle behavior)
            db.delete(existing_like)
            db.commit()
            return {
                "status": "success",
                "message": "Like removed"
            }
    
    elif interaction.interaction_type == "read":
        # For reads, update existing read interaction or create new one
        existing_read = db.query(Interaction).filter(
            Interaction.user_id == interaction.user_id,
            Interaction.article_id == interaction.article_id,
            Interaction.interaction_type == "read"
        ).first()
        
        if existing_read:
            # Update existing read with new timestamp and read time
            existing_read.timestamp = datetime.now(timezone.utc)
            if interaction.read_time_seconds is not None:
                existing_read.read_time_seconds = interaction.read_time_seconds
            db.commit()
            db.refresh(existing_read)
            
            return {
                "status": "success",
                "message": "Read interaction updated"
            }
    
    # For clicks and skips, always allow multiple (track all instances)
    
    # Create new interaction
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
    topics = db.query(Article.topic, func.count(Article.id))\
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


# =================================================================
# ==================== Classification Endpoints ====================
# =================================================================

@app.post(
    "/api/classify/text",
    response_model=ClassificationResult,
    tags=["classification"],
    summary="Classify raw text",
    description="Classify a piece of text into one of 4 topics using the fine-tuned model"
)
async def classify_text_endpoint(request: ClassifyTextRequest):
    """
    Classify raw text into one of 4 topics.

    Topics:
    - World (0): International news, politics, global events
    - Sports (1): Sports news, games, athletes
    - Business (2): Financial news, markets, companies
    - Sci/Tech (3): Science, technology, innovation

    Returns topic prediction with confidence score.
    """
    from backend.ml import classify_text

    try:
        result = classify_text(
            request.text,
            return_all_scores=request.return_all_scores
        )
        return result
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.post(
    "/api/classify/article/{article_id}",
    response_model=ClassificationResult,
    tags=["classification"],
    summary="Classify article by ID",
    description="Classify an existing article from the database"
)
async def classify_article_endpoint(
    article_id: int,
    return_all_scores: bool = Query(False, description="Return scores for all topics"),
    db: Session = Depends(get_db)
):
    """
    Classify an article that's already in the database.

    Useful for:
    - Testing the classifier on real articles
    - Re-classifying articles
    - Getting classification without saving to database
    """
    from backend.ml import classify_text

    # Get article from database
    article = db.query(Article).filter(Article.id == article_id).first()

    if not article:
        raise HTTPException(
            status_code=404,
            detail=f"Article with ID {article_id} not found"
        )

    if not article.content:
        raise HTTPException(
            status_code=400,
            detail="Article has no content to classify"
        )

    try:
        result = classify_text(
            article.content,
            return_all_scores=return_all_scores
        )
        return result
    except Exception as e:
        logger.error(f"Classification error for article {article_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.post(
    "/api/classify/batch",
    response_model=BatchClassificationResponse,
    tags=["classification"],
    summary="Classify multiple texts",
    description="Classify multiple texts efficiently in a single request"
)
async def classify_batch_endpoint(request: BatchClassifyRequest):
    """
    Classify multiple texts in batch for better performance.

    - Up to 100 texts per request
    - Processes texts in batches of 16 for efficiency
    - Returns results in the same order as input
    """
    from backend.ml import get_classifier
    import time

    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="No texts provided"
        )

    try:
        classifier = get_classifier()

        start_time = time.time()
        results = classifier.classify_batch(request.texts, batch_size=16)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "results": results,
            "total": len(results),
            "processing_time_ms": round(elapsed, 2)
        }
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )


@app.get(
    "/api/classify/model-info",
    tags=["classification"],
    summary="Get model information",
    description="Get information about the loaded classification model"
)
async def get_model_info():
    """
    Get metadata about the classification model.

    Returns:
    - Model status (loaded/not loaded)
    - Model type
    - Available topics
    - Device (CPU/GPU)
    """
    from backend.ml import get_classifier

    try:
        classifier = get_classifier()
        info = classifier.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


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