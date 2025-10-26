# News Recommender System

A personalized news recommendation platform with ML-powered topic classification and RAG-based Q&A.

## Features
- Topic classification using fine-tuned DistilBERT (94.92% accuracy)
- Personalized recommendations (collaborative filtering + content-based)
- RAG system for natural language Q&A
- Real-time article processing
- RESTful API

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB RAM

### Setup
```bash
# Clone this repo and create new venv
cd personal-news-recommender
python -m venv venv
. venv/bin/activate && pip install -r requirements.txt

# Configure environment
# Edit .env with your settings

# Start services
docker-compose up -d

# Initialize database
python scripts/init_db.py

# Run the app (in different terminals)
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
celery -A backend.celery_app worker --pool=solo --loglevel=info -Q celery,processing,scraping
celery -A backend.celery_app beat --loglevel=info
