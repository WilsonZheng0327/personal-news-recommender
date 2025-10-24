# Setup Guide - Article Processing Pipeline

Complete guide to set up and run the article processing pipeline for the News Recommender system.

## Table of Contents
1. [Quick Start with Docker](#quick-start-with-docker)
2. [Manual Setup](#manual-setup)
3. [Database Setup](#database-setup)
4. [Testing the Pipeline](#testing-the-pipeline)
5. [Running with Celery](#running-with-celery)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start with Docker

The fastest way to get PostgreSQL and Redis running is using Docker Compose.

### 1. Start Services

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

You should see:
```
NAME                IMAGE                  STATUS
news_postgres       postgres:15-alpine     Up (healthy)
news_redis          redis:7-alpine         Up (healthy)
```

### 2. Verify Connections

**Test PostgreSQL:**
```bash
docker exec -it news_postgres psql -U admin -d news_recommender -c "SELECT 1;"
```

**Test Redis:**
```bash
docker exec -it news_redis redis-cli ping
# Should return: PONG
```

### 3. Environment Variables

Make sure your `.env` file has these values (matching docker-compose.yaml):

```bash
# Database (matches docker-compose postgres service)
DATABASE_URL=postgresql://admin:password@localhost:5432/news_recommender

# Redis (matches docker-compose redis service)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### 4. Data Persistence

Data is stored in the `./data` directory:
- `./data/postgres/` - PostgreSQL database files
- `./data/redis/` - Redis persistence files
- `./data/faiss_index.bin` - FAISS vector index (created by processing pipeline)

### 5. Stop Services

```bash
# Stop but keep data
docker-compose down

# Stop and remove all data (CAUTION: Deletes everything!)
docker-compose down -v
rm -rf data/
```

---

## Manual Setup

If you prefer not to use Docker, here's how to set up PostgreSQL and Redis manually.

### PostgreSQL Setup

**Windows:**
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer, set password for `postgres` user
3. Create database:
   ```bash
   psql -U postgres
   CREATE DATABASE news_recommender;
   CREATE USER admin WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE news_recommender TO admin;
   \q
   ```

**macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15

createdb news_recommender
```

**Linux:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

sudo -u postgres psql
CREATE DATABASE news_recommender;
```

### Redis Setup

**Windows:**
- Use Memurai: https://www.memurai.com/get-memurai
- Or WSL: `wsl -e sudo service redis-server start`

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Test Redis:**
```bash
redis-cli ping
# Should return: PONG
```

---

## Database Setup

### 1. Install Python Dependencies

Make sure you have Python 3.10+ and all packages installed:

```bash
pip install -r requirements.txt
```

### 2. Initialize Database Tables

Create all tables (articles, users, interactions):

```bash
python backend/db/init_db.py
```

Expected output:
```
Creating database tables...
SUCCESS: Database initialized successfully!
```

### 3. Verify Tables

```bash
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article
db = SessionLocal()
print(f'Articles table exists: {db.query(Article).count() >= 0}')
db.close()
"
```

### 4. Scrape Test Data

Get some articles to process:

```bash
python backend/scrapers/rss_scraper.py
```

This scrapes 50-100 articles from various news sources (TechCrunch, Wired, Reddit, etc.)

Expected output:
```
🔄 Starting scrape of 13 feeds...

Scraping: https://www.theguardian.com/world/rss
  ✓ Added: UN holds emergency meeting on Middle East crisis...
  ✓ Added: Climate summit reaches historic agreement...
  📊 Summary - Added: 15 | Skipped: 2 | Failed: 0

...

✅ Scraping complete in 142.3 seconds!
📰 Total articles in database: 87
```

---

## Testing the Pipeline

Before setting up Celery, test the processing pipeline manually to ensure everything works.

### Step 1: Manual Processing Script

Process articles WITHOUT Celery (useful for debugging):

```bash
# Process all pending articles
python scripts/process_articles_manual.py

# Process in smaller batches
python scripts/process_articles_manual.py --batch-size 10

# Process a specific article
python scripts/process_articles_manual.py --article-id 1
```

**What this does:**
1. ✅ Loads the fine-tuned DistilBERT classifier
2. ✅ Loads the sentence-transformers embedder
3. ✅ Queries pending articles from database
4. ✅ Classifies each article into topics (World, Sports, Business, Sci/Tech)
5. ✅ Generates 384-dimensional embeddings
6. ✅ Adds vectors to FAISS index for similarity search
7. ✅ Updates database with topic, confidence, embedding_id
8. ✅ Saves FAISS index to disk

**Expected output:**
```
======================================================================
Batch Processing (max 50 articles)
======================================================================

Querying for pending articles...
Found 87 pending articles

Loading ML models...
Using device: cpu
Model loaded successfully in 3.24s
  Classifier: {'status': 'loaded', 'model_type': 'distilbert', 'num_labels': 4}
Model loaded successfully in 1.82s
  Embedder: {'status': 'loaded', 'dimension': 384}
  Vector store: {'total_vectors': 0, 'dimension': 384, 'index_type': 'flat'}

Processing articles...
----------------------------------------------------------------------

[1/87] Article 1
  Title: Tesla announces new affordable electric vehicle for 2025...
  Topic: Sci/Tech (confidence: 0.956)

[2/87] Article 2
  Title: NBA Finals: Lakers defeat Celtics in Game 7 thriller...
  Topic: Sports (confidence: 0.982)

[3/87] Article 3
  Title: Federal Reserve raises interest rates to combat inflation...
  Topic: Business (confidence: 0.891)

...

Generating embeddings for 87 articles...
100%|███████████████████████████████| 87/87 [00:12<00:00,  7.11it/s]
  Embeddings shape: (87, 384)

Adding 87 vectors to FAISS index...

Updating articles in database...

Saving FAISS index to data/faiss_index.bin...

======================================================================
📊 Processing Summary
======================================================================
  Total processed: 87
  Failed: 0
  Success rate: 100.0%

📈 Overall Database Stats
  Total articles: 87
  Pending: 0
  Completed: 87
  Failed: 0

🔍 FAISS Index Stats
  total_vectors: 87
  dimension: 384
  index_type: flat
  is_trained: True
======================================================================
```

### Step 2: Verify Processing Results

Check that articles were processed correctly:

```bash
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article

db = SessionLocal()

# Get stats
total = db.query(Article).count()
pending = db.query(Article).filter(Article.processing_status == 'pending').count()
completed = db.query(Article).filter(Article.processing_status == 'completed').count()
failed = db.query(Article).filter(Article.processing_status == 'failed').count()

print(f'📊 Processing Stats:')
print(f'  Total articles: {total}')
print(f'  Pending: {pending}')
print(f'  Completed: {completed}')
print(f'  Failed: {failed}')

# Show sample processed article
article = db.query(Article).filter(Article.processing_status == 'completed').first()
if article:
    print(f'\n📰 Sample Processed Article:')
    print(f'  ID: {article.id}')
    print(f'  Title: {article.title[:60]}...')
    print(f'  Topic: {article.topic}')
    print(f'  Confidence: {article.topic_confidence:.3f}')
    print(f'  Embedding ID: {article.embedding_id}')
    print(f'  Processed at: {article.processed_at}')

db.close()
"
```

### Step 3: Test Topic Classification Distribution

See how articles were classified:

```bash
python -c "
from backend.db.database import SessionLocal
from backend.db.models import Article
from sqlalchemy import func

db = SessionLocal()

# Get topic distribution
topics = db.query(Article.topic, func.count(Article.id))\
    .filter(Article.topic.isnot(None))\
    .group_by(Article.topic)\
    .all()

print('📈 Topic Distribution:')
for topic, count in topics:
    print(f'  {topic}: {count}')

db.close()
"
```

---

## Running with Celery

Once manual processing works, set up Celery for automatic background processing.

### Architecture Overview

```
┌─────────────┐
│   FastAPI   │  Main application (port 8000)
│     API     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Redis    │  Message broker (port 6379)
│   (Queue)   │  Stores tasks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Celery    │  Background workers
│   Worker    │  Process tasks from queue
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ PostgreSQL  │  Database (port 5432)
│   (Data)    │  Store results
└─────────────┘
```

### Step 1: Start Services

Make sure PostgreSQL and Redis are running:

```bash
# With Docker
docker-compose up -d

# Verify
docker-compose ps
```

### Step 2: Start Celery Worker

Open a **new terminal** and run:

```bash
# Windows
celery -A backend.celery_app worker --loglevel=info --pool=solo

# macOS/Linux
celery -A backend.celery_app worker --loglevel=info
```

**Why `--pool=solo` on Windows?**
- Windows doesn't support fork(), which Celery uses by default
- `--pool=solo` uses single-threaded execution (safe for Windows)

**Expected output:**
```
 -------------- celery@YourComputer v5.3.0
---- **** -----
--- * ***  * -- Windows-10-... 2025-01-20 10:00:00
-- * - **** ---
- ** ---------- [config]
- ** ---------- .> app:         news_recommender:0x...
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/0
- ** ---------- .> concurrency: 1 (solo)
- *** --- * --- .> task events: OFF
-- ******* ---- .> task routes:
--- ***** -----      backend.tasks.processing_tasks.* -> processing
 -------------- [queues]
                .> processing       exchange=processing(direct) key=processing

[tasks]
  . backend.tasks.processing_tasks.get_processing_stats
  . backend.tasks.processing_tasks.process_pending_articles
  . backend.tasks.processing_tasks.process_single_article
  . backend.tasks.processing_tasks.reprocess_failed_articles
  . backend.tasks.processing_tasks.save_faiss_index

[2025-01-20 10:00:00,123: INFO/MainProcess] Connected to redis://localhost:6379/0
[2025-01-20 10:00:00,456: INFO/MainProcess] mingle: searching for neighbors
[2025-01-20 10:00:01,789: INFO/MainProcess] mingle: all alone
[2025-01-20 10:00:02,012: INFO/MainProcess] celery@YourComputer ready.
```

✅ If you see "celery@YourComputer ready", the worker is running!

### Step 3: Start Celery Beat (Optional - For Automatic Scheduling)

For automatic periodic processing, open **another terminal**:

```bash
celery -A backend.celery_app beat --loglevel=info
```

This scheduler will automatically trigger:
- `process_pending_articles` every 10 minutes
- `save_faiss_index` every hour

**Expected output:**
```
celery beat v5.3.0 is starting.
__    -    ... __   -        _
LocalTime -> 2025-01-20 10:00:00
Configuration ->
    . broker -> redis://localhost:6379/0
    . loader -> celery.loaders.app.AppLoader
    . scheduler -> celery.beat.PersistentScheduler
    . db -> celerybeat-schedule
    . logfile -> [stderr]@%INFO
    . maxinterval -> 5.00 minutes (300s)
[2025-01-20 10:00:00,123: INFO/MainProcess] beat: Starting...
[2025-01-20 10:00:00,456: INFO/MainProcess] Scheduler: Sending due task process-pending-articles
```

### Step 4: Manually Trigger Tasks

**Option 1: Python Shell**

```python
from backend.tasks.processing_tasks import process_pending_articles

# Trigger task
task = process_pending_articles.delay()
print(f"Task ID: {task.id}")

# Check status (blocking - waits for result)
result = task.get(timeout=300)
print(result)
```

**Option 2: Command Line**

```bash
python -c "
from backend.tasks.processing_tasks import process_pending_articles
task = process_pending_articles.delay()
print(f'Task submitted: {task.id}')
print('Check worker terminal to see processing...')
"
```

**Option 3: Via FastAPI (coming soon)**

We'll add API endpoints like:
- `POST /api/process/trigger` - Manually trigger processing
- `GET /api/process/stats` - Get processing statistics
- `POST /api/process/article/{id}` - Process single article

### Step 5: Monitor Flower (Web UI)

Flower provides a beautiful web interface for monitoring Celery.

```bash
# Start Flower
celery -A backend.celery_app flower
```

Open browser to: **http://localhost:5555**

You'll see:
- 📊 Dashboard with task statistics
- 👷 Active workers and their status
- 📋 Task history and details
- 📈 Real-time graphs
- ⚙️ Worker configuration

---

## Troubleshooting

### Redis Connection Error

**Error:**
```
redis.exceptions.ConnectionError: Error 10061 connecting to localhost:6379
```

**Solutions:**
1. Make sure Redis is running:
   ```bash
   # Docker
   docker-compose ps

   # Manual
   redis-cli ping
   ```

2. Check Redis URL in `.env` matches your setup:
   ```bash
   REDIS_URL=redis://localhost:6379/0
   ```

3. On Windows: Use Docker or Memurai (not native Redis)

### Model Not Found Error

**Error:**
```
FileNotFoundError: Model not found at models/topic-classifier
```

**Solutions:**
1. Check `TOPIC_CLASSIFIER_PATH` in `.env`:
   ```bash
   TOPIC_CLASSIFIER_PATH=models/topic-classifier
   ```

2. Make sure you've trained the model (see training notebooks)

3. Use absolute path if needed:
   ```bash
   TOPIC_CLASSIFIER_PATH=C:/Users/yourname/path/to/models/topic-classifier
   ```

### Database Connection Error

**Error:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions:**
1. Check PostgreSQL is running:
   ```bash
   # Docker
   docker-compose ps

   # Manual
   pg_isready
   ```

2. Verify `DATABASE_URL` in `.env`:
   ```bash
   # Docker setup
   DATABASE_URL=postgresql://admin:password@localhost:5432/news_recommender
   ```

3. Test connection:
   ```bash
   psql -h localhost -U admin -d news_recommender
   ```

### Celery Worker Won't Start

**Error:**
```
celery.exceptions.ImproperlyConfigured
```

**Solutions:**
1. On Windows, **always use** `--pool=solo`:
   ```bash
   celery -A backend.celery_app worker --loglevel=info --pool=solo
   ```

2. Make sure Redis is running **before** starting worker

3. Check for typos: `backend.celery_app` (not `celery_app`)

4. Verify Python can import:
   ```bash
   python -c "from backend.celery_app import celery_app; print('OK')"
   ```

### Out of Memory Error

**Error:** Worker crashes or system freezes

**Solutions:**
1. Reduce batch size in `.env`:
   ```bash
   PROCESSING_BATCH_SIZE=10  # Instead of 50
   ```

2. Process in smaller chunks:
   ```bash
   python scripts/process_articles_manual.py --batch-size 10
   ```

3. The worker config already limits memory:
   ```python
   worker_max_tasks_per_child=50  # Restart after 50 tasks
   worker_prefetch_multiplier=1   # Process one at a time
   ```

### FAISS Index Corrupted

**Error:**
```
RuntimeError: Error in faiss::read_index
```

**Solutions:**
1. Delete corrupted index files:
   ```bash
   rm data/faiss_index.bin data/faiss_index.pkl
   ```

2. Rebuild index by reprocessing:
   ```sql
   UPDATE articles SET processing_status = 'pending', embedding_id = NULL;
   ```

3. Reprocess all articles:
   ```bash
   python scripts/process_articles_manual.py
   ```

### No Articles to Process

**Issue:** Script says "No pending articles found"

**Solutions:**
1. Scrape some articles first:
   ```bash
   python backend/scrapers/rss_scraper.py
   ```

2. Check article statuses:
   ```bash
   python -c "
   from backend.db.database import SessionLocal
   from backend.db.models import Article
   db = SessionLocal()
   print(f'Total: {db.query(Article).count()}')
   print(f'Pending: {db.query(Article).filter(Article.processing_status == \"pending\").count()}')
   db.close()
   "
   ```

3. Reset processed articles to pending (if needed):
   ```bash
   python -c "
   from backend.db.database import SessionLocal
   from backend.db.models import Article
   db = SessionLocal()
   db.query(Article).update({'processing_status': 'pending'})
   db.commit()
   print('Reset all articles to pending')
   db.close()
   "
   ```

---

## Next Steps

Once your processing pipeline is working:

1. ✅ **Add API endpoints** for triggering processing via FastAPI
2. ✅ **Build recommendation engine** using processed articles
3. ✅ **Implement RAG system** for conversational Q&A
4. ✅ **Add monitoring** with custom metrics and alerts
5. ✅ **Optimize performance** with GPU inference, caching, batching

**Congratulations!** Your article processing pipeline is now set up and running! 🎉

For usage examples and API documentation, see [usage-guide.md](./usage-guide.md).
