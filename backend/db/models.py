from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

'''
When creating a class inheriting from Base,
SQLAlchemy automatically adds this table to Base.metadata
'''
class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text)
    source = Column(String)
    published_at = Column(DateTime)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # ML fields
    topic = Column(String, nullable=True)  # Will be filled by classifier
    topic_confidence = Column(Float, nullable=True)
    embedding_id = Column(Integer, nullable=True)  # FAISS index position

    # Processing status tracking
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    processed_at = Column(DateTime, nullable=True)  # When processing completed
    processing_error = Column(Text, nullable=True)  # Error message if failed
    
    # Metadata
    author = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    
    # Relationship for convenience (optional - can remove if prefer manual queries)
    interactions = relationship("Interaction", back_populates="article")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    interactions = relationship("Interaction", back_populates="user")

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    article_id = Column(Integer, ForeignKey("articles.id"))
    interaction_type = Column(String)  # 'click', 'read', 'like', 'skip'
    timestamp = Column(DateTime, default=datetime.utcnow)
    read_time_seconds = Column(Integer, nullable=True)
    
    user = relationship("User", back_populates="interactions")
    article = relationship("Article", back_populates="interactions")