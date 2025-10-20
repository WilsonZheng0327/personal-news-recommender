from pydantic_settings import BaseSettings
from functools import lru_cache

"""
BaseSettings automatically loads from .env files
- case insensitive
- checks if var exists in .env
- if not, use default value set here
"""

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "news-recommender"
    env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Database
    database_url: str
    db_echo: bool = False       # SQLAlchemy prints all SQL queries if True
    
    # Redis
    redis_url: str
    
    # Celery
    celery_broker_url: str      # where Celery gets tasks from
    celery_result_backend: str  # where Celery stores task results
    
    # API
    api_host: str = "0.0.0.0"   # listen on all network interfaces
                                # compared to "127.0.0.1" only localhost
                                # (can't access from other machines)
    api_port: int = 8000        # has auto type conversion
    api_reload: bool = True     # auto-reload on code changes
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # ML Models
    topic_classifier_path: str
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss_index"
    
    # Subclass that tells Pydantic to load from .env file
    class Config:
        env_file = ".env"
        case_sensitive = False

# Saved to cache after first call
@lru_cache()
def get_settings() -> Settings:
    """Cache settings for performance"""
    return Settings()