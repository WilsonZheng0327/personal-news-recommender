# Article Processing Pipeline - Quick Start

This README provides a quick overview and getting started guide for the article processing pipeline.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Start infrastructure (PostgreSQL + Redis)
docker-compose up -d

# 2. Initialize database
python backend/db/init_db.py

# 3. Scrape some articles (optional)
python backend/scrapers/rss_scraper.py

# 4. Process articles
python scripts/process_articles_manual.py
```

That's it! Your articles are now classified and embedded.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [design_docs/setup-guide.md](design_docs/setup-guide.md) | Complete setup instructions (Docker, Redis, PostgreSQL, Celery) |
| [design_docs/usage-guide.md](design_docs/usage-guide.md) | Usage examples, API integration, workflows |
| [design_docs/pipeline-summary.md](design_docs/pipeline-summary.md) | Implementation overview and architecture |
| [design_docs/architecture.md](design_docs/architecture.md) | Full system architecture |

---

## 🏗️ What This Pipeline Does

The article processing pipeline:

1. ✅ **Queries** pending articles from PostgreSQL
2. ✅ **Classifies** them into topics (World, Sports, Business, Sci/Tech)
3. ✅ **Generates** 384-dimensional semantic embeddings
4. ✅ **Indexes** vectors in FAISS for fast similarity search
5. ✅ **Updates** database with results
6. ✅ **Saves** FAISS index to disk

**Processing Flow:**
```
[Articles in DB] → [Classifier] → [Embedder] → [FAISS Index] → [Ready for Recommendations]
     (pending)      (DistilBERT)  (all-MiniLM)    (384-dim)          (completed)
```

---

## 🛠️ Two Ways to Run

### Option 1: Manual Script (Recommended for Testing)

No Celery needed - great for debugging:

```bash
python scripts/process_articles_manual.py

# Options
python scripts/process_articles_manual.py --batch-size 20
python scripts/process_articles_manual.py --article-id 42
```

### Option 2: Celery Background Tasks (Production)

Automatic background processing with scheduling:

```bash
# Terminal 1: Start worker
celery -A backend.celery_app worker --loglevel=info --pool=solo

# Terminal 2: Start scheduler (optional)
celery -A backend.celery_app beat --loglevel=info

# Terminal 3: Trigger processing
python -c "from backend.tasks.processing_tasks import process_pending_articles; process_pending_articles.delay()"
```

---

## 📊 Check Processing Results

```bash
# Quick stats
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article
db = SessionLocal()
print(f'Total: {db.query(Article).count()}')
print(f'Completed: {db.query(Article).filter(Article.processing_status == \"completed\").count()}')
db.close()
"

# Detailed stats
python -c "from backend.tasks.processing_tasks import get_processing_stats; print(get_processing_stats.delay().get())"
```

---

## ⚙️ Configuration

Edit `.env` file to configure:

```bash
# Database
DATABASE_URL=postgresql://admin:password@localhost:5432/news_recommender

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# ML Models
TOPIC_CLASSIFIER_PATH=models/topic-classifier
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index.bin

# Processing
PROCESSING_BATCH_SIZE=50
PROCESSING_INTERVAL_MINUTES=10
```

---

## 🐛 Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Redis connection error | `docker-compose up -d` |
| No articles to process | `python backend/scrapers/rss_scraper.py` |
| Model not found | Check `TOPIC_CLASSIFIER_PATH` in `.env` |
| Celery won't start (Windows) | Use `--pool=solo` flag |

See [design_docs/setup-guide.md](design_docs/setup-guide.md#troubleshooting) for detailed troubleshooting.

---

## 📁 Key Files

```
backend/
├── celery_app.py              # Celery configuration
├── tasks/
│   └── processing_tasks.py    # Background tasks
└── db/
    └── models.py              # Article model with processing fields

scripts/
└── process_articles_manual.py # Manual processing script

design_docs/
├── setup-guide.md            # Setup instructions
├── usage-guide.md            # Usage examples
└── pipeline-summary.md       # Implementation overview
```

---

## 🎯 Next Steps

After the pipeline is working, implement:

1. **Recommendation Engine** - Use FAISS for content-based recommendations
2. **RAG System** - Conversational Q&A over articles
3. **API Endpoints** - Trigger processing via FastAPI
4. **Monitoring** - Add Flower for task monitoring

See [design_docs/pipeline-summary.md](design_docs/pipeline-summary.md#whats-next) for the full roadmap.

---

## 🚨 Important Notes

- **Windows Users:** Always use `--pool=solo` flag with Celery
- **First Run:** Models take 2-5 seconds to load (they're cached after)
- **Processing Speed:** ~2-5 articles/second on CPU, ~20-50 on GPU
- **Memory:** Each worker loads models (~1-2GB RAM), don't run too many
- **Data Persistence:** All data stored in `./data/` directory

---

## 📞 Getting Help

1. Check documentation: [design_docs/](design_docs/)
2. View logs: Celery worker terminal shows detailed processing logs
3. Flower monitoring: `celery -A backend.celery_app flower` → http://localhost:5555
4. Database inspection: Use any PostgreSQL client to query the database

---

**Status:** ✅ Ready to use

**Version:** 1.0 (Initial implementation)

**Last Updated:** 2025-01-20
