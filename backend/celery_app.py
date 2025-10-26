"""
Celery Application Configuration

This module sets up Celery for background task processing.

Architecture:
- Celery workers pick up tasks from Redis queue
- Tasks run asynchronously (don't block API)
- Results stored in Redis for tracking
- Beat scheduler triggers periodic tasks

Usage:
    # Start worker (run this in a terminal)
    celery -A backend.celery_app worker --loglevel=info --pool=solo

    # Start beat scheduler (for periodic tasks)
    celery -A backend.celery_app beat --loglevel=info

    # Monitor tasks with Flower (optional)
    celery -A backend.celery_app flower
"""

from celery import Celery
from celery.schedules import crontab
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "news_recommender",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "backend.tasks.processing_tasks",  # Import task modules
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,    # Acknowledge task after completion (safer)
                            # if False, task stuck forever if fails
    task_reject_on_worker_lost=True,  # Requeue if worker crashes

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store additional metadata

    # Worker settings
    worker_prefetch_multiplier=1,   # Process one task at a time (safer for ML models)
                                    # how many tasks a worker picks up each time
                                    # and stores in local queue (not available to others)
    worker_max_tasks_per_child=10,  # Restart worker after 50 tasks (prevent memory leaks)

    # Retry settings
    task_autoretry_for=(Exception,),  # Auto-retry on any exception
    task_retry_kwargs={'max_retries': 3, 'countdown': 60},  # Retry up to 3 times, wait 60s

    # Logging
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

# Periodic tasks (Celery Beat schedule)
celery_app.conf.beat_schedule = {
    # Process unclassified articles every 10 minutes
    "process-pending-articles": {
        "task": "backend.tasks.processing_tasks.process_pending_articles",
        "schedule": crontab(minute=f"*/{settings.processing_interval_minutes}"),  # Every N minutes
        "options": {"queue": "processing"}
    },

    # Save FAISS index to disk every hour (backup)
    "save-faiss-index": {
        "task": "backend.tasks.processing_tasks.save_faiss_index",
        "schedule": crontab(minute=0),  # Every hour at :00
        "options": {"queue": "processing"}
    },

    # Test task (uses default queue for simplicity)
    "test-task": {
        "task": "backend.tasks.processing_tasks.test_task",
        "schedule": crontab(minute='*'),
    }
}

# Task routing (optional - route different tasks to different queues)
celery_app.conf.task_routes = {
    "backend.tasks.processing_tasks.process_pending_articles": {"queue": "processing"},
    "backend.tasks.processing_tasks.process_single_article": {"queue": "processing"},
    "backend.tasks.processing_tasks.save_faiss_index": {"queue": "processing"},
    "backend.tasks.processing_tasks.reprocess_failed_articles": {"queue": "processing"},
    "backend.tasks.processing_tasks.get_processing_stats": {"queue": "processing"},
    # test_task uses default queue
    "backend.tasks.scraping_tasks.*": {"queue": "scraping"},  # For future scraping tasks
}

logger.info("Celery app configured successfully")

if __name__ == "__main__":
    # For debugging
    print("Celery app configuration:")
    print(f"  Broker: {celery_app.conf.broker_url}")
    print(f"  Backend: {celery_app.conf.result_backend}")
    print(f"  Task modules: {celery_app.conf.include}")
    print(f"  Beat schedule: {list(celery_app.conf.beat_schedule.keys())}")
