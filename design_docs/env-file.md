# .env File Configuration - Complete Guide

## The Big Picture: Why Different Environments Matter

```
Your Code → Reads .env → Connects to services

Local Development:     localhost:5432
Staging Server:        staging-db.company.com:5432
Production Server:     prod-db.company.com:5432

Same code, different .env files!
```
---

## Quick Reference: What to Change When Deploying

| Setting | Development | Production |
|---------|-------------|------------|
| **ENV** | `development` | `production` |
| **DEBUG** | `True` | `False` |
| **DATABASE_URL** | `localhost:5432` | Cloud hostname |
| **REDIS_URL** | `localhost:6379` | Cloud hostname |
| **SECRET_KEY** | Simple string | Strong random key |
| **API_RELOAD** | `True` | `False` |
| **LOG_LEVEL** | `INFO` or `DEBUG` | `WARNING` |
| **Model Paths** | Same! | Same! |


---

## Your .env File for Local Development

### Complete .env (Copy this!)

```bash
# ============================================================================
# LOCAL DEVELOPMENT ENVIRONMENT
# ============================================================================

# Application
APP_NAME=news-recommender
ENV=development
DEBUG=True
LOG_LEVEL=INFO

# Database (localhost because you're running Docker locally)
DATABASE_URL=postgresql://news_user:news_password_dev@localhost:5432/news_recommender
DB_ECHO=False

# Redis (localhost because you're running Docker locally)
REDIS_URL=redis://localhost:6379/0

# Celery (localhost because you're running Docker locally)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Security (CHANGE THIS!)
SECRET_KEY=dev-secret-key-change-this-in-production-use-openssl-rand-hex-32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Models (local paths)
TOPIC_CLASSIFIER_PATH=backend/ml/models/topic_classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index

# External APIs (add when needed)
# OPENAI_API_KEY=sk-your-key-here
# NEWS_API_KEY=your-news-api-key
```

---

## Understanding "localhost" vs Service Names

### The Confusing Part Explained:

**Two contexts where you connect to databases:**

#### Context 1: Your Code Running on Your Computer (OUTSIDE Docker)
```python
# You run: python backend/api/main.py
# This runs ON YOUR COMPUTER, not in Docker

DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
#                                       ↑
#                                  Use "localhost"!
```

**Why localhost?** 
- Your Python code is running on your computer
- Docker mapped port 5432 to your computer
- So you connect to localhost:5432

#### Context 2: Your Code Running Inside a Docker Container
```python
# Later when you containerize your app:
# docker-compose.yml has your FastAPI app as a service

DATABASE_URL = "postgresql://user:pass@postgres:5432/db"
#                                       ↑
#                                  Use service name!
```

**Why service name?**
- Your app is inside Docker network
- Other containers are accessible by service name
- Can't use "localhost" (that means the container itself!)

---

## Visual Diagram: localhost vs Service Names

### Current Setup (App NOT containerized):
```
┌─────────────────────────────────────┐
│     Your Computer (macOS/Linux)     │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Your Python App             │   │
│  │ (runs on computer)          │   │
│  │                             │   │
│  │ DATABASE_URL=               │   │
│  │   localhost:5432 ←──────┐  │   │
│  └─────────────────────────┼───┘   │
│                            │       │
│  ┌─────────────────────────┼───┐   │
│  │ Docker                  │   │   │
│  │                         ↓   │   │
│  │  ┌─────────────────────────┐│   │
│  │  │ postgres:5432           ││   │
│  │  │ (port mapped to         ││   │
│  │  │  host 5432)             ││   │
│  │  └─────────────────────────┘│   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

Connection: localhost:5432 → Docker's port mapping → postgres:5432
```

### Future Setup (App IN Docker):
```
┌─────────────────────────────────────┐
│     Your Computer                   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Docker                      │   │
│  │                             │   │
│  │  ┌──────────────────────┐   │   │
│  │  │ Your FastAPI App     │   │   │
│  │  │                      │   │   │
│  │  │ DATABASE_URL=        │   │   │
│  │  │   postgres:5432 ─────┼───┼───┐
│  │  └──────────────────────┘   │   │
│  │                             │   │
│  │  ┌──────────────────────┐   │   │
│  │  │ postgres:5432    ←───────┼───┘
│  │  │                      │   │
│  │  └──────────────────────┘   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

Connection: postgres:5432 → Direct within Docker network
```

---

## When to Change from localhost

### Scenario 1: You Keep App Outside Docker (Current)
```bash
# .env stays with localhost
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Why?
# - Your app runs on your computer
# - Docker containers expose ports to localhost
# - Never needs to change!
```

**Keep using localhost for:**
- ✅ Running FastAPI with `uvicorn` directly
- ✅ Running Celery workers on your machine
- ✅ Running tests locally
- ✅ Development scripts

### Scenario 2: You Containerize Your App (Future)
```bash
# Create .env.docker
DATABASE_URL=postgresql://user:pass@postgres:5432/db
REDIS_URL=redis://redis:6379/0
#                    ↑
#              Service names, not localhost!
```

**Add to docker-compose.yml:**
```yaml
services:
  app:
    build: .
    container_name: news_app
    env_file:
      - .env.docker  # Use different env file!
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    networks:
      - news_network
```

### Scenario 3: Deploy to Production
```bash
# .env.production
DATABASE_URL=postgresql://user:pass@prod-db-server.aws.com:5432/db
REDIS_URL=redis://prod-redis.aws.com:6379/0
#                  ↑
#           Real server addresses!
```

---

## The Three .env Files You'll Have

### 1. `.env` (Local development - what you use now)
```bash
# Running on your computer, Docker services on localhost
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
DEBUG=True
ENV=development
```

### 2. `.env.docker` (When you containerize app)
```bash
# Everything running in Docker, use service names
DATABASE_URL=postgresql://user:pass@postgres:5432/db
REDIS_URL=redis://redis:6379/0
DEBUG=True
ENV=development
```

### 3. `.env.production` (Deploy to real server)
```bash
# Real production database servers
DATABASE_URL=postgresql://prod_user:strong_pass@prod-db.example.com:5432/news_recommender
REDIS_URL=redis://prod-redis.example.com:6379/0
DEBUG=False
ENV=production
SECRET_KEY=super-secret-key-from-secrets-manager
```

---

## Complete Example: All Three Files

### `.env` (Use this now!)
```bash
# ============================================================================
# LOCAL DEVELOPMENT - App runs on computer, connects to Docker via localhost
# ============================================================================

APP_NAME=news-recommender
ENV=development
DEBUG=True
LOG_LEVEL=DEBUG

# Use localhost - Docker maps ports to your computer
DATABASE_URL=postgresql://news_user:news_password_dev@localhost:5432/news_recommender
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Weak password OK for local dev
SECRET_KEY=local-dev-secret-not-for-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Local paths
TOPIC_CLASSIFIER_PATH=backend/ml/models/topic_classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index

# External APIs (optional for local dev)
# OPENAI_API_KEY=sk-test-key
```

### `.env.docker` (For later when you add app to docker-compose)
```bash
# ============================================================================
# DOCKER DEVELOPMENT - Everything in Docker, use service names
# ============================================================================

APP_NAME=news-recommender
ENV=development
DEBUG=True
LOG_LEVEL=DEBUG

# Use service names - containers talk within Docker network
DATABASE_URL=postgresql://news_user:news_password_dev@postgres:5432/news_recommender
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

SECRET_KEY=docker-dev-secret-not-for-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Paths inside container
TOPIC_CLASSIFIER_PATH=/app/backend/ml/models/topic_classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=/app/data/faiss_index

# OPENAI_API_KEY=sk-test-key
```

### `.env.production` (For deployment)
```bash
# ============================================================================
# PRODUCTION - Real servers, real security
# ============================================================================

APP_NAME=news-recommender
ENV=production
DEBUG=False
LOG_LEVEL=WARNING

# Production database (managed service like AWS RDS)
DATABASE_URL=postgresql://prod_user:${DB_PASSWORD}@prod-db.cxyz123.us-east-1.rds.amazonaws.com:5432/news_recommender

# Production Redis (managed service like AWS ElastiCache)
REDIS_URL=redis://prod-redis.abc123.0001.use1.cache.amazonaws.com:6379/0
CELERY_BROKER_URL=redis://prod-redis.abc123.0001.use1.cache.amazonaws.com:6379/0
CELERY_RESULT_BACKEND=redis://prod-redis.abc123.0001.use1.cache.amazonaws.com:6379/1

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=False

# Strong secret from AWS Secrets Manager or similar
SECRET_KEY=${SECRET_KEY_FROM_SECRETS_MANAGER}
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15

# Production paths
TOPIC_CLASSIFIER_PATH=/app/models/topic_classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=/app/data/faiss_index

# Real API keys
OPENAI_API_KEY=${OPENAI_API_KEY_FROM_SECRETS}
NEWS_API_KEY=${NEWS_API_KEY_FROM_SECRETS}

# Production-specific
ALLOWED_ORIGINS=https://news-recommender.com,https://www.news-recommender.com
SENTRY_DSN=${SENTRY_DSN}  # Error tracking
```

---

## How to Use Different .env Files

### Method 1: Manual Switching
```bash
# For local development
cp .env.example .env
# Edit .env with localhost URLs

# When containerizing
cp .env.docker .env
# Now uses service names
```

### Method 2: Specify File (Better)
```bash
# Load specific env file
python -m backend.api.main --env-file .env

# Or in docker-compose.yml
services:
  app:
    env_file:
      - .env.docker  # Explicitly specify which file
```

### Method 3: Environment-based (Best)
```python
# config/settings.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "news-recommender"
    env: str = "development"
    
    # Automatically use correct host based on environment
    @property
    def db_host(self):
        if os.getenv("RUNNING_IN_DOCKER"):
            return "postgres"  # Service name
        return "localhost"  # Local development
    
    @property
    def database_url(self):
        return f"postgresql://user:pass@{self.db_host}:5432/db"
    
    class Config:
        env_file = ".env"
```

Then set in docker-compose.yml:
```yaml
services:
  app:
    environment:
      - RUNNING_IN_DOCKER=true
```

---

## Common Questions

### Q: "Do I change localhost to something else when deploying?"

**A: Yes!**

```bash
# Local development
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Deploy to Railway
DATABASE_URL=postgresql://user:pass@containers-us-west-123.railway.app:5432/db

# Deploy to AWS
DATABASE_URL=postgresql://user:pass@my-db.cxyz.us-east-1.rds.amazonaws.com:5432/db

# Deploy with Docker Compose on server
DATABASE_URL=postgresql://user:pass@postgres:5432/db
```

### Q: "When do I use localhost vs service names?"

**A: Depends where your app runs:**

| Your App Runs | Database Connection |
|---------------|---------------------|
| On your computer (outside Docker) | `localhost:5432` |
| Inside Docker (same docker-compose) | `postgres:5432` |
| On Railway/Heroku/AWS | Provider's URL |

### Q: "How do I know if I need to change it?"

**A: Test your connection!**

```python
# Create scripts/test_env.py
from config.settings import get_settings

settings = get_settings()
print(f"Environment: {settings.env}")
print(f"Database URL: {settings.database_url}")
print(f"Redis URL: {settings.redis_url}")

# Try connecting
from backend.db.database import engine
try:
    with engine.connect() as conn:
        print("✅ Database connection works!")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("💡 Check your DATABASE_URL in .env")
```

### Q: "What about the SECRET_KEY?"

**Generate a secure one:**
```bash
# Generate random secret key
openssl rand -hex 32

# Output: 8f7d9a6b3c2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6

# Add to .env
SECRET_KEY=8f7d9a6b3c2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6
```

**Never use the same secret in different environments!**

---

## .gitignore for .env Files

```gitignore
# Environment files (contains secrets!)
.env
.env.local
.env.*.local

# Keep examples (no secrets)
!.env.example

# Production configs (never commit!)
.env.production
.env.staging
```

---

## Migration Path: Development → Production

### Stage 1: Local Development (Now)
```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/db
```

Run: `uvicorn backend.api.main:app --reload`

### Stage 2: Containerize Locally
```bash
# .env.docker
DATABASE_URL=postgresql://user:pass@postgres:5432/db
```

Run: `docker-compose up`

### Stage 3: Deploy to Railway/Heroku
```bash
# Railway provides DATABASE_URL automatically
# Just use their environment variables
```

Railway dashboard: Set environment variables

### Stage 4: Deploy to AWS
```bash
# .env.production
DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/db
```

Store in AWS Secrets Manager or Parameter Store

---

## Quick Start: Create Your .env Now

```bash
# 1. Copy example
cp .env.example .env

# 2. Generate secret key
SECRET_KEY=$(openssl rand -hex 32)

# 3. Edit .env (use localhost for now!)
cat > .env << EOF
APP_NAME=news-recommender
ENV=development
DEBUG=True
LOG_LEVEL=INFO

DATABASE_URL=postgresql://news_user:news_password_dev@localhost:5432/news_recommender
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

SECRET_KEY=$SECRET_KEY
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

TOPIC_CLASSIFIER_PATH=backend/ml/models/topic_classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index
EOF

# 4. Test it
python scripts/test_connection.py
```

---

## Summary

### Right Now (Local Development):
```bash
# ✅ Use localhost
DATABASE_URL=postgresql://...@localhost:5432/...
REDIS_URL=redis://localhost:6379/0
```

**Why?** Your code runs on your computer, Docker exposes ports to localhost

### Later (When Containerizing):
```bash
# ✅ Use service names
DATABASE_URL=postgresql://...@postgres:5432/...
REDIS_URL=redis://redis:6379/0
```

**Why?** Everything in Docker, containers talk via service names

### Production (When Deploying):
```bash
# ✅ Use real server URLs
DATABASE_URL=postgresql://...@prod-server.com:5432/...
REDIS_URL=redis://prod-redis.com:6379/0
```

**Why?** Connecting to actual hosted databases

**Bottom line**: Start with localhost, change when you deploy. Your code doesn't change, just the .env file! 🎯

