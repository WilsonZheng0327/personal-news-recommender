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
# Clone and setup
git clone <your-repo>
cd news-recommender
make setup

# Configure environment
# Edit .env with your settings

# Start services
make start

# Initialize database
python scripts/init_db.py

# Run the app (in another terminal)
source venv/bin/activate
uvicorn backend.api.main:app --reload