"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

# ==================== Article Schemas ====================

class ArticleResponse(BaseModel):
    """Response schema for article list"""
    id: int
    title: str
    url: str
    source: str
    topic: str | None = None
    published_at: datetime | None = None
    
    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy models


class ArticleDetailResponse(BaseModel):
    """Response schema for single article with full details"""
    id: int
    title: str
    content: str
    url: str
    source: str
    topic: str | None = None
    topic_confidence: float | None = None
    author: str | None = None
    image_url: str | None = None
    published_at: datetime | None = None
    scraped_at: datetime
    
    class Config:
        from_attributes = True


# ==================== Interaction Schemas ====================

class InteractionCreate(BaseModel):
    """Request schema for creating an interaction"""
    user_id: int = Field(..., description="ID of the user", ge=1)
    article_id: int = Field(..., description="ID of the article", ge=1)
    interaction_type: Literal["click", "read", "like", "skip"] = Field(
        ..., 
        description="Type of interaction"
    )
    read_time_seconds: int | None = Field(
        None, 
        description="Time spent reading (in seconds)", 
        ge=0
    )


class InteractionResponse(BaseModel):
    """Response schema for interaction"""
    id: int
    user_id: int
    article_id: int
    interaction_type: str
    timestamp: datetime
    read_time_seconds: int | None = None
    
    class Config:
        from_attributes = True


# ==================== User Schemas ====================

class UserCreate(BaseModel):
    """Request schema for creating a user"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")


class UserResponse(BaseModel):
    """Response schema for user"""
    id: int
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== Generic Response Schemas ====================

class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = "success"
    message: str | None = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str | None = None
    timestamp: datetime | None = None


class CountResponse(BaseModel):
    """Response with count"""
    count: int


# ==================== Classification Schemas ====================

class ClassifyTextRequest(BaseModel):
    """Request schema for classifying raw text"""
    text: str = Field(
        ...,
        description="Text to classify",
        min_length=1,
        max_length=10000
    )
    return_all_scores: bool = Field(
        False,
        description="If True, return confidence scores for all topics"
    )


class ClassifyArticleRequest(BaseModel):
    """Request schema for classifying an article by ID"""
    article_id: int = Field(..., description="ID of the article to classify", ge=1)
    return_all_scores: bool = Field(
        False,
        description="If True, return confidence scores for all topics"
    )


class ClassificationResult(BaseModel):
    """Response schema for classification result"""
    topic: str = Field(..., description="Predicted topic")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    topic_id: int = Field(..., description="Topic ID (0-3)")
    all_scores: dict[str, float] | None = Field(
        None,
        description="Confidence scores for all topics (if requested)"
    )


class BatchClassifyRequest(BaseModel):
    """Request schema for batch classification"""
    texts: list[str] = Field(
        ...,
        description="List of texts to classify",
        min_length=1,
        max_length=100
    )


class BatchClassificationResponse(BaseModel):
    """Response schema for batch classification"""
    results: list[ClassificationResult]
    total: int
    processing_time_ms: float