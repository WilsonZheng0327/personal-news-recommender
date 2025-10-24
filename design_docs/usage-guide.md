# Usage Guide - Article Processing Pipeline

This guide explains how to use the article processing pipeline in different scenarios.

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Manual Processing](#manual-processing)
3. [Celery Tasks](#celery-tasks)
4. [API Integration](#api-integration)
5. [Common Workflows](#common-workflows)
6. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Quick Reference

### Start Everything

```bash
# 1. Start PostgreSQL and Redis
docker-compose up -d

# 2. Start Celery worker (in separate terminal)
celery -A backend.celery_app worker --loglevel=info --pool=solo

# 3. Start Celery beat for scheduled tasks (optional, in another terminal)
celery -A backend.celery_app beat --loglevel=info

# 4. Start FastAPI server (in another terminal)
python backend/api/main.py
```

### Stop Everything

```bash
# Stop Celery (Ctrl+C in their terminals)
# Stop Docker services
docker-compose down
```

### Quick Processing

```bash
# Process all pending articles (no Celery needed)
python scripts/process_articles_manual.py

# Process with Celery
python -c "from backend.tasks.processing_tasks import process_pending_articles; process_pending_articles.delay()"
```

---

## Manual Processing

For one-time processing, testing, or when you don't want to run Celery.

### Process All Pending Articles

```bash
python scripts/process_articles_manual.py
```

This will:
- Load ML models (classifier, embedder)
- Query articles with `processing_status = 'pending'`
- Classify topics for all articles
- Generate embeddings in batch
- Add vectors to FAISS index
- Update database with results
- Save FAISS index to disk

**Options:**

```bash
# Process only 20 articles
python scripts/process_articles_manual.py --batch-size 20

# Process a specific article by ID
python scripts/process_articles_manual.py --article-id 42
```

### Process After Scraping

```bash
# 1. Scrape articles
python backend/scrapers/rss_scraper.py

# 2. Process them immediately
python scripts/process_articles_manual.py
```

### Reprocess Failed Articles

```bash
# Reset failed articles to pending
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article

db = SessionLocal()
failed = db.query(Article).filter(Article.processing_status == 'failed').all()
for article in failed:
    article.processing_status = 'pending'
    article.processing_error = None
db.commit()
print(f'Reset {len(failed)} failed articles')
db.close()
"

# Then process
python scripts/process_articles_manual.py
```

### Update Existing Articles

If you want to reclassify already processed articles:

```bash
# Reset all to pending
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article

db = SessionLocal()
db.query(Article).update({
    'processing_status': 'pending',
    'topic': None,
    'topic_confidence': None,
    'embedding_id': None
})
db.commit()
db.close()
"

# Delete old FAISS index
rm data/faiss_index.bin data/faiss_index.pkl

# Reprocess everything
python scripts/process_articles_manual.py
```

---

## Celery Tasks

For background processing with automatic scheduling.

### Available Tasks

| Task | Description | Triggered By |
|------|-------------|--------------|
| `process_pending_articles` | Process all pending articles in batch | Beat (every 10 min), Manual |
| `process_single_article` | Process one article by ID | Manual, API |
| `save_faiss_index` | Save FAISS index to disk | Beat (hourly), After processing |
| `reprocess_failed_articles` | Reset failed articles to pending | Manual |
| `get_processing_stats` | Get article processing statistics | Manual, API |

### Trigger Tasks Manually

**Python Shell:**

```python
from backend.tasks.processing_tasks import (
    process_pending_articles,
    process_single_article,
    get_processing_stats,
    reprocess_failed_articles,
    save_faiss_index
)

# Process all pending
task = process_pending_articles.delay()
print(f"Task ID: {task.id}")

# Wait for result (blocking)
result = task.get(timeout=300)  # 5 minute timeout
print(result)
# {
#     'status': 'success',
#     'processed': 42,
#     'failed': 0,
#     'total': 42,
#     'timestamp': '2025-01-20T10:30:00'
# }

# Process single article
task = process_single_article.delay(article_id=123)
result = task.get(timeout=60)
print(result)
# {
#     'status': 'success',
#     'article_id': 123,
#     'topic': 'Sci/Tech',
#     'confidence': 0.956,
#     'embedding_id': 42
# }

# Get stats (non-blocking)
task = get_processing_stats.delay()
# Don't wait, check later
print(f"Task submitted: {task.id}")

# Reprocess failed articles
task = reprocess_failed_articles.delay()
result = task.get()
print(result)
# {
#     'status': 'success',
#     'reset_count': 5
# }
```

**Command Line:**

```bash
# Process all pending
python -c "from backend.tasks.processing_tasks import process_pending_articles; print(process_pending_articles.delay().id)"

# Process single article
python -c "from backend.tasks.processing_tasks import process_single_article; process_single_article.delay(42)"

# Get stats
python -c "from backend.tasks.processing_tasks import get_processing_stats; import time; task = get_processing_stats.delay(); time.sleep(1); print(task.result)"
```

### Check Task Status

```python
from backend.celery_app import celery_app

# Get task by ID
task_id = "abc-123-def-456"
result = celery_app.AsyncResult(task_id)

print(f"State: {result.state}")  # PENDING, STARTED, SUCCESS, FAILURE
print(f"Info: {result.info}")    # Task result or error info

if result.successful():
    print(f"Result: {result.result}")
elif result.failed():
    print(f"Error: {result.info}")
```

### Cancel Running Task

```python
from backend.celery_app import celery_app

task_id = "abc-123-def-456"
celery_app.control.revoke(task_id, terminate=True)
```

### Scheduled Tasks (Celery Beat)

Tasks that run automatically when Beat is running:

```python
# In backend/celery_app.py

celery_app.conf.beat_schedule = {
    "process-pending-articles": {
        "task": "backend.tasks.processing_tasks.process_pending_articles",
        "schedule": crontab(minute="*/10"),  # Every 10 minutes
    },
    "save-faiss-index": {
        "task": "backend.tasks.processing_tasks.save_faiss_index",
        "schedule": crontab(minute=0),  # Every hour at :00
    },
}
```

**Change Schedule:**

Edit `.env`:
```bash
PROCESSING_INTERVAL_MINUTES=5  # Process every 5 minutes instead of 10
```

Then restart Celery worker and beat.

---

## API Integration

Add endpoints to your FastAPI app to trigger processing via HTTP requests.

### Example Endpoints (To Be Implemented)

**1. Trigger Processing**

```python
# backend/api/main.py

from backend.tasks.processing_tasks import process_pending_articles

@app.post("/api/process/trigger")
async def trigger_processing():
    """Trigger article processing"""
    task = process_pending_articles.delay()
    return {
        "status": "processing",
        "task_id": task.id,
        "message": "Processing started in background"
    }
```

**Usage:**
```bash
curl -X POST http://localhost:8000/api/process/trigger
```

**2. Get Processing Stats**

```python
from backend.tasks.processing_tasks import get_processing_stats

@app.get("/api/process/stats")
async def processing_stats():
    """Get article processing statistics"""
    task = get_processing_stats.delay()
    result = task.get(timeout=10)
    return result
```

**Usage:**
```bash
curl http://localhost:8000/api/process/stats
```

**Response:**
```json
{
  "articles": {
    "total": 100,
    "pending": 10,
    "processing": 2,
    "completed": 85,
    "failed": 3
  },
  "faiss": {
    "total_vectors": 85,
    "dimension": 384,
    "index_type": "flat"
  },
  "timestamp": "2025-01-20T10:30:00"
}
```

**3. Process Single Article**

```python
from backend.tasks.processing_tasks import process_single_article

@app.post("/api/process/article/{article_id}")
async def process_article(article_id: int):
    """Process a specific article"""
    task = process_single_article.delay(article_id)
    return {
        "status": "processing",
        "task_id": task.id,
        "article_id": article_id
    }
```

**Usage:**
```bash
curl -X POST http://localhost:8000/api/process/article/42
```

**4. Check Task Status**

```python
from backend.celery_app import celery_app

@app.get("/api/process/task/{task_id}")
async def check_task_status(task_id: str):
    """Check status of a processing task"""
    result = celery_app.AsyncResult(task_id)

    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info,
        "successful": result.successful(),
        "failed": result.failed()
    }
```

**Usage:**
```bash
curl http://localhost:8000/api/process/task/abc-123-def-456
```

---

## Common Workflows

### Workflow 1: Continuous News Processing

**Goal:** Automatically scrape and process news articles every hour.

**Setup:**

1. Create scraping task (future work):
```python
# backend/tasks/scraping_tasks.py

@celery_app.task
def scrape_and_process():
    """Scrape articles then process them"""
    # Run scraper
    from backend.scrapers.rss_scraper import scrape_all_feeds
    scrape_all_feeds()

    # Trigger processing
    process_pending_articles.delay()
```

2. Add to beat schedule:
```python
celery_app.conf.beat_schedule["scrape-and-process"] = {
    "task": "backend.tasks.scraping_tasks.scrape_and_process",
    "schedule": crontab(minute=0),  # Every hour
}
```

3. Start services:
```bash
docker-compose up -d
celery -A backend.celery_app worker --loglevel=info --pool=solo
celery -A backend.celery_app beat --loglevel=info
```

### Workflow 2: Manual Article Upload & Processing

**Goal:** User uploads article via API, process it immediately.

```python
# backend/api/main.py

@app.post("/api/articles/upload")
async def upload_article(
    title: str,
    content: str,
    source: str,
    db: Session = Depends(get_db)
):
    """Upload a new article and process it"""
    from backend.tasks.processing_tasks import process_single_article

    # Create article
    article = Article(
        url=f"manual/{datetime.now().timestamp()}",
        title=title,
        content=content,
        source=source,
        published_at=datetime.now(timezone.utc),
        processing_status="pending"
    )
    db.add(article)
    db.commit()
    db.refresh(article)

    # Process it
    task = process_single_article.delay(article.id)

    return {
        "article_id": article.id,
        "task_id": task.id,
        "message": "Article created and processing started"
    }
```

### Workflow 3: Batch Reprocessing After Model Update

**Goal:** You've fine-tuned a better classifier, reprocess all articles.

```bash
# 1. Save new model
# (After training, save to models/topic-classifier/)

# 2. Update .env to point to new model
TOPIC_CLASSIFIER_PATH=models/topic-classifier-v2

# 3. Restart Celery worker to reload model
# Ctrl+C then restart

# 4. Reset all articles to pending
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article
db = SessionLocal()
count = db.query(Article).update({'processing_status': 'pending'})
db.commit()
print(f'Reset {count} articles')
db.close()
"

# 5. Process in batches (to avoid overload)
for i in {1..10}; do
    python scripts/process_articles_manual.py --batch-size 100
    sleep 60  # Wait 1 minute between batches
done
```

### Workflow 4: Monitoring & Error Recovery

**Goal:** Monitor processing and handle failures.

```bash
# 1. Check processing stats
python -c "
from backend.tasks.processing_tasks import get_processing_stats
result = get_processing_stats.delay().get()
print(result)
"

# 2. View failed articles
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article
db = SessionLocal()
failed = db.query(Article).filter(Article.processing_status == 'failed').all()
for article in failed:
    print(f'Article {article.id}: {article.processing_error}')
db.close()
"

# 3. Retry failed articles
python -c "
from backend.tasks.processing_tasks import reprocess_failed_articles
result = reprocess_failed_articles.delay().get()
print(result)
"

# 4. Monitor with Flower
celery -A backend.celery_app flower
# Open http://localhost:5555
```

---

## Monitoring and Debugging

### View Celery Worker Logs

The worker terminal shows real-time logs:

```
[2025-01-20 10:30:00,123: INFO/MainProcess] Task backend.tasks.processing_tasks.process_pending_articles[abc-123] received
[2025-01-20 10:30:00,456: INFO/MainProcess] Starting batch article processing...
[2025-01-20 10:30:02,789: INFO/MainProcess] Found 42 pending articles
[2025-01-20 10:30:05,012: INFO/MainProcess] Generating embeddings for 42 articles...
[2025-01-20 10:30:15,345: INFO/MainProcess] Batch processing complete: {'status': 'success', 'processed': 42, 'failed': 0}
[2025-01-20 10:30:15,678: INFO/MainProcess] Task backend.tasks.processing_tasks.process_pending_articles[abc-123] succeeded
```

### Check FAISS Index Status

```python
from backend.ml.vector_store import get_vector_store

store = get_vector_store()
stats = store.get_stats()
print(stats)
# {
#     'total_vectors': 87,
#     'dimension': 384,
#     'index_type': 'flat',
#     'id_map_size': 87,
#     'is_trained': True
# }

# Test search
import numpy as np
query = np.random.randn(384).astype(np.float32)
article_ids, distances = store.search(query, k=5)
print(f"Similar articles: {article_ids}")
```

### Check Model Status

```python
from backend.ml import get_classifier, get_embedder

# Classifier
classifier = get_classifier()
print(classifier.get_model_info())
# {
#     'status': 'loaded',
#     'model_type': 'distilbert',
#     'num_labels': 4,
#     'topics': ['World', 'Sports', 'Business', 'Sci/Tech'],
#     'device': 'cpu',
#     'max_length': 512
# }

# Embedder
embedder = get_embedder()
print(embedder.get_model_info())
# {
#     'status': 'loaded',
#     'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
#     'dimension': 384,
#     'max_seq_length': 256
# }
```

### Database Queries

```python
from backend.db.database import SessionLocal
from backend.db.models import Article
from sqlalchemy import func

db = SessionLocal()

# Articles by status
for status in ['pending', 'processing', 'completed', 'failed']:
    count = db.query(Article).filter(Article.processing_status == status).count()
    print(f"{status}: {count}")

# Recent processed articles
recent = db.query(Article)\
    .filter(Article.processing_status == 'completed')\
    .order_by(Article.processed_at.desc())\
    .limit(5)\
    .all()

for article in recent:
    print(f"{article.id}: {article.title[:50]}... [{article.topic}]")

# Topic distribution
topics = db.query(Article.topic, func.count(Article.id))\
    .filter(Article.topic.isnot(None))\
    .group_by(Article.topic)\
    .all()

for topic, count in topics:
    print(f"{topic}: {count}")

# Average confidence by topic
avg_conf = db.query(
    Article.topic,
    func.avg(Article.topic_confidence)
)\
    .filter(Article.topic.isnot(None))\
    .group_by(Article.topic)\
    .all()

for topic, avg in avg_conf:
    print(f"{topic}: {avg:.3f}")

db.close()
```

### Flower Monitoring

Start Flower for web-based monitoring:

```bash
celery -A backend.celery_app flower
```

Open **http://localhost:5555** and you can:

- 📊 See task success/failure rates
- ⏱️ View task execution times
- 👷 Monitor active workers
- 📋 Browse task history
- 🔍 Search for specific tasks
- 📈 View real-time graphs
- ⚙️ Inspect worker configuration
- 🚫 Revoke/cancel tasks

### Redis Monitoring

```bash
# Connect to Redis CLI
docker exec -it news_redis redis-cli

# Check queue length
LLEN celery

# View keys
KEYS *

# Monitor commands in real-time
MONITOR
```

### Debug Mode

For verbose logging:

```bash
# Worker with debug logging
celery -A backend.celery_app worker --loglevel=debug --pool=solo

# Beat with debug logging
celery -A backend.celery_app beat --loglevel=debug
```

---

## Performance Tips

### Batch Processing

Always process in batches for better performance:

```python
# ❌ Slow (one by one)
for article_id in article_ids:
    process_single_article.delay(article_id)

# ✅ Fast (batch)
process_pending_articles.delay()  # Processes up to 50 at once
```

### GPU Acceleration

If you have a GPU, the models will automatically use it:

```python
# Check device
from backend.ml import get_classifier
classifier = get_classifier()
print(classifier.device)  # cuda or cpu
```

Processing speed:
- **CPU**: ~2-5 articles/second
- **GPU**: ~20-50 articles/second

### Parallel Workers

Run multiple workers for faster processing:

```bash
# Terminal 1
celery -A backend.celery_app worker --loglevel=info --pool=solo -n worker1@%h

# Terminal 2
celery -A backend.celery_app worker --loglevel=info --pool=solo -n worker2@%h

# Terminal 3
celery -A backend.celery_app worker --loglevel=info --pool=solo -n worker3@%h
```

**Note:** Each worker loads ML models into memory, so don't run too many!

### Memory Optimization

Workers restart after 50 tasks to prevent memory leaks:

```python
# backend/celery_app.py
celery_app.conf.worker_max_tasks_per_child = 50
```

Lower this if you experience memory issues:

```python
worker_max_tasks_per_child = 20
```

---

## Summary

**Quick Commands:**

```bash
# Manual processing (no Celery)
python scripts/process_articles_manual.py

# Start worker
celery -A backend.celery_app worker --loglevel=info --pool=solo

# Start beat
celery -A backend.celery_app beat --loglevel=info

# Start monitoring
celery -A backend.celery_app flower

# Trigger processing
python -c "from backend.tasks.processing_tasks import process_pending_articles; process_pending_articles.delay()"

# Get stats
python -c "from backend.tasks.processing_tasks import get_processing_stats; print(get_processing_stats.delay().get())"
```

For setup instructions, see [setup-guide.md](./setup-guide.md).

For architecture details, see [architecture.md](./architecture.md).
