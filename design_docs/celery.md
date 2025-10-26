# Complete Guide to Celery

## What is Celery?

**Celery is a distributed task queue system** that lets you run tasks in the background, outside of your main application.

### The Problem It Solves

**Without Celery:**
```python
@app.post("/api/scrape-articles")
async def scrape_articles():
    # This blocks the API request!
    articles = scrape_rss_feeds()  # Takes 5 minutes
    for article in articles:
        classify(article)           # Takes 10 seconds each
        embed(article)              # Takes 5 seconds each

    return {"status": "done"}  # User waits 20+ minutes! 😱
```

**With Celery:**
```python
@app.post("/api/scrape-articles")
async def scrape_articles():
    # Queue the task, return immediately!
    task = scrape_articles_task.delay()
    return {"task_id": task.id, "status": "queued"}  # Returns in <1ms ✅

# Task runs in the background
@celery_app.task
def scrape_articles_task():
    articles = scrape_rss_feeds()  # Runs in background worker
    # ... process articles ...
```

---

## How Celery Works (Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR CELERY SYSTEM                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   FastAPI    │        │    Redis     │        │    Celery    │
│   Server     │───────▶│   (Broker)   │───────▶│   Worker     │
│              │  send  │   [Queue]    │  pull  │              │
└──────────────┘  task  └──────────────┘  task  └──────────────┘
                           ▲                           │
                           │         store             │
                           │         result            │
                           │                           ▼
                        ┌─────────────────────────────┐
                        │     Redis (Backend)         │
                        │     [Task Results]          │
                        └─────────────────────────────┘

Optional:
┌──────────────┐                          ┌──────────────┐
│ Celery Beat  │───schedule tasks────────▶│    Redis     │
│ (Scheduler)  │                          │   (Broker)   │
└──────────────┘                          └──────────────┘
```

### Components:

**1. Broker (Redis)**
- Message queue that holds tasks
- FastAPI puts tasks here
- Workers pull tasks from here

**2. Workers**
- Separate processes that execute tasks
- Can run multiple workers for parallel processing
- Load your ML models and execute the tasks

**3. Backend (Redis)**
- Stores task results
- Tracks task status (pending/running/success/failed)
- Optional (can run without it)

**4. Beat Scheduler (Optional)**
- Cron-like scheduler
- Triggers periodic tasks automatically
- Example: "Process articles every 10 minutes"

**5. Flower (Optional)**
- Web UI to monitor tasks
- See which tasks are running, failed, etc.
- Access at `http://localhost:5555`

---

## Your Celery Configuration Explained

### Section 1: Basic Setup

```python
celery_app = Celery(
    "news_recommender",                      # App name
    broker=settings.celery_broker_url,       # Redis URL for queue
    backend=settings.celery_result_backend,  # Redis URL for results
    include=[
        "backend.tasks.processing_tasks",    # Where your tasks are defined
    ]
)
```

**What this does:**
- Creates a Celery application named "news_recommender"
- Connects to Redis for the message queue (broker)
- Connects to Redis for storing results (backend)
- Tells Celery where to find task definitions

**Your .env values:**
```bash
CELERY_BROKER_URL=redis://localhost:6379/0      # Database 0 for queue
CELERY_RESULT_BACKEND=redis://localhost:6379/1  # Database 1 for results
```

---

### Section 2: Task Settings

```python
task_serializer="json",       # Encode tasks as JSON
accept_content=["json"],      # Only accept JSON tasks (security)
result_serializer="json",     # Encode results as JSON
timezone="UTC",               # Use UTC timezone
enable_utc=True,             # Enable UTC support
```

**Why JSON?**
- Cross-language compatible
- Secure (prevents arbitrary code execution)
- Easy to debug

**Alternative:** `pickle` (faster but less secure)

---

### Section 3: Task Execution (IMPORTANT!)

```python
task_acks_late=True,                # ← IMPORTANT!
task_reject_on_worker_lost=True,    # ← IMPORTANT!
```

**What is "acknowledgement"?**

When a worker pulls a task from the queue:

**With `task_acks_late=False` (default):**
```
1. Worker: "I'm taking task #123" (acks immediately)
2. Redis: *removes task from queue*
3. Worker: *starts processing*
4. Worker: *CRASHES* 💥
5. Task is LOST forever! ❌
```

**With `task_acks_late=True` (safer):**
```
1. Worker: *takes task #123 but doesn't ack yet*
2. Redis: *keeps task in queue as "processing"*
3. Worker: *starts processing*
4. Worker: *CRASHES* 💥
5. Redis: "Worker died, re-queue the task" ♻️
6. Another worker picks it up ✅
```

**When to use:**
- ✅ Use `task_acks_late=True` for important tasks (your ML processing)
- ❌ Use `False` for disposable tasks (sending emails)

**`task_reject_on_worker_lost=True`:**
- If worker dies, put task back in queue
- Works with `task_acks_late=True`

---

### Section 4: Result Backend

```python
result_expires=3600,    # Delete results after 1 hour
result_extended=True,   # Store extra metadata
```

**What are results?**

```python
# You call a task
task = process_article.delay(article_id=42)

# Later, check the result
result = task.get(timeout=10)  # Wait up to 10s
# result = {"classified": True, "topic": "Sports", "embedded": True}
```

**Result lifecycle:**
```
Task starts    → result_expires (1 hour) → Result deleted
```

**Why expire?**
- Redis memory is limited
- Old results aren't needed
- Prevents memory bloat

---

### Section 5: Worker Settings

```python
worker_prefetch_multiplier=1,       # Process 1 task at a time
worker_max_tasks_per_child=50,     # Restart worker after 50 tasks
```

**`worker_prefetch_multiplier=1`:**

Controls how many tasks a worker grabs from the queue at once.

**With prefetch=4 (default):**
```
Worker 1: *grabs 4 tasks*
Tasks: [Task A, Task B, Task C, Task D]
Processing: Task A (takes 10 minutes)
Tasks B, C, D: *waiting in worker's local queue*

Problem: If Worker 1 is slow, Tasks B/C/D can't be stolen by other workers!
```

**With prefetch=1 (your setting):**
```
Worker 1: *grabs 1 task*
Task: [Task A]
Processing: Task A

Worker 2: *grabs Task B from Redis*
Worker 3: *grabs Task C from Redis*

Better load distribution! ✅
```

**Why prefetch=1 for ML tasks:**
- ML models use lots of memory
- Tasks take variable time
- Better load balancing across workers
- Prevents one worker from hogging all tasks

**`worker_max_tasks_per_child=50`:**

Restarts worker process after 50 tasks.

**Why?**
- **Memory leaks**: Python/ML models can leak memory
- **Fresh start**: Reload models periodically
- **Prevent crashes**: Better than having worker die randomly

**Tradeoff:**
- ✅ Prevents memory issues
- ⚠️ Slight overhead restarting
- ⚠️ Reloads ML models (but they're singletons, so cached)

---

### Section 6: Retry Settings

```python
task_autoretry_for=(Exception,),                        # Retry on any error
task_retry_kwargs={'max_retries': 3, 'countdown': 60},  # 3 retries, 60s wait
```

**What happens when a task fails?**

**Without auto-retry:**
```
Task: classify_article(id=42)
Error: TimeoutError (network issue)
Status: FAILED ❌
```

**With auto-retry:**
```
Attempt 1: classify_article(id=42)
Error: TimeoutError
Wait 60 seconds...

Attempt 2: classify_article(id=42)
Error: TimeoutError
Wait 60 seconds...

Attempt 3: classify_article(id=42)
Success! ✅

Attempt 4 (if failed): Mark as FAILED ❌
```

**Why this is good:**
- Transient errors (network glitches) often resolve
- Don't lose tasks due to temporary issues
- Automatic recovery

**Countdown (backoff):**
- Wait 60 seconds between retries
- Gives time for issue to resolve
- Prevents hammering a failing service

---

### Section 7: Periodic Tasks

```python
celery_app.conf.beat_schedule = {
    "process-pending-articles": {
        "task": "backend.tasks.processing_tasks.process_pending_articles",
        "schedule": crontab(minute=f"*/{settings.processing_interval_minutes}"),
        "options": {"queue": "processing"}
    },
}
```

**This is like a cron job in Celery!**

See [crontab.md](crontab.md) for detailed crontab syntax.

---

### Section 8: Task Routing

```python
celery_app.conf.task_routes = {
    "backend.tasks.processing_tasks.*": {"queue": "processing"},
    "backend.tasks.scraping_tasks.*": {"queue": "scraping"},
}
```

**What are queues?**

You can have multiple queues for different types of tasks:

```
Redis Broker:
┌──────────────────────────────────┐
│  Queue: "processing"             │
│  ├── classify_article(id=1)      │
│  ├── embed_article(id=2)         │
│  └── process_batch(ids=[3,4,5])  │
├──────────────────────────────────┤
│  Queue: "scraping"               │
│  ├── scrape_rss_feed(url=...)    │
│  └── scrape_article(url=...)     │
└──────────────────────────────────┘
```

**Why use multiple queues?**

**Scenario 1: Different priorities**
```bash
# High priority worker (process 100 tasks/hour)
celery -A backend.celery_app worker -Q processing --concurrency=4

# Low priority worker (process 10 tasks/hour)
celery -A backend.celery_app worker -Q scraping --concurrency=1
```

**Scenario 2: Different machines**
```bash
# On GPU machine
celery -A backend.celery_app worker -Q processing

# On CPU machine (scraping doesn't need GPU)
celery -A backend.celery_app worker -Q scraping
```

**Scenario 3: Isolation**
```
If scraping tasks crash → processing tasks keep running ✅
```

---

## Real-World Example: Your System

### Scenario: User Triggers Article Processing

**Step 1: API Request**
```python
# User hits endpoint
POST /api/process-articles

# FastAPI handler
@app.post("/api/process-articles")
async def process_articles():
    task = process_pending_articles.delay()  # Queue the task
    return {"task_id": task.id, "status": "queued"}
```

**Step 2: Task Goes to Redis Queue**
```
Redis (Broker):
queue: "processing"
├── Task ID: abc123
│   ├── name: "process_pending_articles"
│   ├── args: []
│   └── status: "PENDING"
```

**Step 3: Worker Picks Up Task**
```bash
# Worker running in terminal:
celery -A backend.celery_app worker

# Output:
[2025-10-25 14:30:00] Received task: process_pending_articles[abc123]
[2025-10-25 14:30:00] Task started
```

**Step 4: Task Executes**
```python
# In worker process
def process_pending_articles():
    articles = db.query(Article).filter(topic=None).all()

    for article in articles:
        # Classify
        result = classify_text(article.content)
        article.topic = result['topic']

        # Embed
        embedding = embed_text(article.content)
        vector_store.add_vector(embedding, article.id)

    db.commit()
    vector_store.save()

    return {"processed": len(articles)}
```

**Step 5: Result Stored**
```
Redis (Backend):
task: "abc123"
├── status: "SUCCESS"
├── result: {"processed": 42}
└── expires: 3600s
```

**Step 6: Check Result**
```python
# Later, in your API or admin panel
from backend.celery_app import celery_app

task_result = celery_app.AsyncResult("abc123")
print(task_result.status)  # "SUCCESS"
print(task_result.result)  # {"processed": 42}
```

---

## How to Use Celery (Commands)

### 1. Start Worker

```bash
# Basic
celery -A backend.celery_app worker --loglevel=info

# Windows (needs solo pool)
celery -A backend.celery_app worker --loglevel=info --pool=solo

# With specific queue
celery -A backend.celery_app worker -Q processing --loglevel=info

# Multiple workers (concurrency)
celery -A backend.celery_app worker --concurrency=4
```

**Output you'll see:**
```
-------------- celery@DESKTOP v5.3.0
---- **** -----
--- * ***  * -- Windows-10
-- * - **** ---
- ** ---------- [config]
- ** ---------- .> app:         news_recommender
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/1
- *** --- * --- .> concurrency: 4
-- ******* ----
--- ***** ----- [queues]
 -------------- .> processing    exchange=processing

[tasks]
  . backend.tasks.processing_tasks.process_pending_articles
  . backend.tasks.processing_tasks.save_faiss_index

[2025-10-25 14:00:00] Ready to receive tasks
```

### 2. Start Beat Scheduler

```bash
celery -A backend.celery_app beat --loglevel=info
```

**Output:**
```
LocalTime -> 2025-10-25 14:00:00
Configuration:
    . broker -> redis://localhost:6379/0
    . loader -> celery.loaders.app.AppLoader
    . scheduler -> celery.beat.PersistentScheduler

Scheduler: Sending due task process-pending-articles
Scheduler: Sending due task save-faiss-index
```

### 3. Monitor with Flower (Optional)

```bash
celery -A backend.celery_app flower
```

Open browser: `http://localhost:5555`

**What you'll see:**
- Dashboard with task statistics
- Real-time task monitoring
- Worker status
- Task history
- Graphs and charts

---

## Common Patterns

### Pattern 1: Queue a Task from API

```python
from backend.tasks.processing_tasks import process_article

@app.post("/api/articles/{article_id}/process")
async def process_article_endpoint(article_id: int):
    # Queue the task
    task = process_article.delay(article_id)

    return {
        "task_id": str(task.id),
        "status": "queued"
    }
```

### Pattern 2: Check Task Status

```python
@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    from backend.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)

    return {
        "task_id": task_id,
        "status": task.status,  # PENDING, STARTED, SUCCESS, FAILURE
        "result": task.result if task.ready() else None
    }
```

### Pattern 3: Wait for Task (Blocking)

```python
# Not recommended for API endpoints!
task = process_article.delay(article_id)
result = task.get(timeout=30)  # Wait up to 30 seconds
```

### Pattern 4: Chain Tasks

```python
from celery import chain

# Task 1 → Task 2 → Task 3
workflow = chain(
    scrape_article.s(url),
    classify_article.s(),
    embed_article.s()
)
workflow.apply_async()
```

---

## Troubleshooting

### Worker Not Starting?

**Check Redis is running:**
```bash
docker ps  # See if Redis container is up
```

**Test Redis connection:**
```python
import redis
r = redis.from_url("redis://localhost:6379/0")
r.ping()  # Should return True
```

### Tasks Not Processing?

**Check worker is running:**
```bash
# Should see celery worker process
ps aux | grep celery
```

**Check queue:**
```python
from backend.celery_app import celery_app
inspect = celery_app.control.inspect()
print(inspect.active())  # See active tasks
print(inspect.reserved())  # See queued tasks
```

### Memory Issues?

Reduce `worker_max_tasks_per_child`:
```python
worker_max_tasks_per_child=10  # Restart after 10 tasks
```

---

## Summary

**Celery is your background job system:**

| Component | What It Does | Example |
|-----------|--------------|---------|
| **Broker (Redis)** | Task queue | Holds pending tasks |
| **Worker** | Executes tasks | Runs your ML code |
| **Backend (Redis)** | Stores results | Task success/failure |
| **Beat** | Scheduler | Runs tasks periodically |
| **Flower** | Monitor | Web UI dashboard |

**Your Configuration Choices:**
- ✅ `task_acks_late=True` - Safe task handling
- ✅ `prefetch=1` - Good for ML tasks
- ✅ `max_tasks_per_child=50` - Prevents memory leaks
- ✅ Auto-retry - Handles transient errors
- ✅ Periodic tasks - Automated processing

**Commands to Remember:**
```bash
# Start worker
celery -A backend.celery_app worker --pool=solo

# Start scheduler
celery -A backend.celery_app beat

# Monitor
celery -A backend.celery_app flower
```