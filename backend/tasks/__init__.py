"""
Background Tasks Module

Contains all Celery tasks for asynchronous processing.
"""

# Import all tasks to ensure they are registered with Celery
from backend.tasks.processing_tasks import (
    process_pending_articles,
    process_single_article,
    save_faiss_index,
    reprocess_failed_articles,
    get_processing_stats,
    test_task,
)

__all__ = [
    "process_pending_articles",
    "process_single_article",
    "save_faiss_index",
    "reprocess_failed_articles",
    "get_processing_stats",
    "test_task",
]
