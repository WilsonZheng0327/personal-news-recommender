"""
Article Processing Tasks

Celery tasks for classifying articles, generating embeddings, and updating FAISS index.

Tasks:
- process_pending_articles: Main task - processes unclassified articles
- process_single_article: Process one article (for manual triggering)
- save_faiss_index: Periodic backup of FAISS index
"""

from datetime import datetime, timezone
from typing import List, Dict
import logging

from celery import Task
from sqlalchemy.orm import Session

from backend.celery_app import celery_app
from backend.db.database import SessionLocal
from backend.db.models import Article
from backend.ml.classifier import get_classifier
from backend.ml.embedder import get_embedder
from backend.ml.vector_store import get_vector_store
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseTask(Task):
    """Base task with database session management"""
    _db_session = None

    def after_return(self, *args, **kwargs):
        """Close database session after task completes"""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None

    @property
    def db(self) -> Session:
        """Get database session (lazy initialization)"""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session


@celery_app.task(
    name="backend.tasks.processing_tasks.process_pending_articles",
    bind=True,
    base=DatabaseTask,
    max_retries=3,
    default_retry_delay=60
)
def process_pending_articles(self) -> Dict:
    """
    Process all pending articles in batch.

    This is the main processing task that:
    1. Queries for unprocessed articles
    2. Classifies them by topic
    3. Generates embeddings
    4. Adds to FAISS index
    5. Updates database

    Returns:
        Dictionary with processing statistics

    Called by:
        - Celery Beat (every N minutes)
        - Manual trigger via API
    """
    logger.info("Starting batch article processing...")

    try:
        # Query for pending articles
        pending_articles = (
            self.db.query(Article)
            .filter(Article.processing_status == "pending")
            .limit(settings.processing_batch_size)
            .all()
        )

        if not pending_articles:
            logger.info("No pending articles to process")
            return {
                "status": "success",
                "processed": 0,
                "failed": 0,
                "message": "No pending articles"
            }

        logger.info(f"Found {len(pending_articles)} pending articles")

        # Load ML models (singleton - only loads once)
        classifier = get_classifier()
        embedder = get_embedder()
        vector_store = get_vector_store()

        # Process articles
        processed_count = 0
        failed_count = 0

        # Prepare batch data
        article_ids = []
        texts_to_embed = []

        for article in pending_articles:
            try:
                # Mark as processing
                article.processing_status = "processing"
                self.db.commit()

                # 1. Classify topic
                logger.debug(f"Classifying article {article.id}: {article.title[:50]}...")

                # Combine title and content for classification
                classification_text = f"{article.title}\n{article.content[:1000]}"

                result = classifier.classify_text(classification_text)

                article.topic = result["topic"]
                article.topic_confidence = result["confidence"]

                logger.debug(f"  Topic: {result['topic']} (confidence: {result['confidence']:.3f})")

                # 2. Prepare for embedding
                article_ids.append(article.id)
                texts_to_embed.append(article.content)

            except Exception as e:
                logger.error(f"Error processing article {article.id}: {e}")
                article.processing_status = "failed"
                article.processing_error = str(e)
                failed_count += 1
                self.db.commit()

        # 3. Generate embeddings in batch (more efficient)
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} articles...")
            embeddings = embedder.embed_batch(texts_to_embed, show_progress=False)

            # 4. Add to FAISS index
            logger.info(f"Adding {len(embeddings)} vectors to FAISS index...")
            faiss_positions = vector_store.add_vectors(embeddings, article_ids)

            # 5. Update articles with embedding IDs and mark as completed
            for article_id, faiss_position in zip(article_ids, faiss_positions):
                article = self.db.query(Article).filter(Article.id == article_id).first()
                if article:
                    article.embedding_id = faiss_position
                    article.processing_status = "completed"
                    article.processed_at = datetime.now(timezone.utc)
                    article.processing_error = None
                    processed_count += 1

            self.db.commit()

            # 6. Save FAISS index to disk (periodic backup)
            logger.info("Saving FAISS index to disk...")
            vector_store.save(settings.faiss_index_path)

        # Return statistics
        result = {
            "status": "success",
            "processed": processed_count,
            "failed": failed_count,
            "total": len(pending_articles),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Batch processing complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Fatal error in batch processing: {e}")
        # Retry task
        raise self.retry(exc=e)


@celery_app.task(
    name="backend.tasks.processing_tasks.process_single_article",
    bind=True,
    base=DatabaseTask,
    max_retries=3
)
def process_single_article(self, article_id: int) -> Dict:
    """
    Process a single article by ID.

    Useful for:
    - Manual reprocessing
    - Immediate processing of new articles
    - Fixing failed articles

    Args:
        article_id: Database ID of the article to process

    Returns:
        Dictionary with processing result
    """
    logger.info(f"Processing single article: {article_id}")

    try:
        # Get article
        article = self.db.query(Article).filter(Article.id == article_id).first()

        if not article:
            return {
                "status": "error",
                "message": f"Article {article_id} not found"
            }

        # Mark as processing
        article.processing_status = "processing"
        self.db.commit()

        # Load models
        classifier = get_classifier()
        embedder = get_embedder()
        vector_store = get_vector_store()

        # 1. Classify
        classification_text = f"{article.title}\n{article.content[:1000]}"
        result = classifier.classify_text(classification_text)

        article.topic = result["topic"]
        article.topic_confidence = result["confidence"]

        # 2. Generate embedding
        embedding = embedder.embed_text(article.content)

        # 3. Add to FAISS
        faiss_position = vector_store.add_vector(embedding, article_id)

        # 4. Update article
        article.embedding_id = faiss_position
        article.processing_status = "completed"
        article.processed_at = datetime.now(timezone.utc)
        article.processing_error = None

        self.db.commit()

        # 5. Save index
        vector_store.save(settings.faiss_index_path)

        return {
            "status": "success",
            "article_id": article_id,
            "topic": article.topic,
            "confidence": article.topic_confidence,
            "embedding_id": faiss_position
        }

    except Exception as e:
        logger.error(f"Error processing article {article_id}: {e}")

        # Mark as failed
        article = self.db.query(Article).filter(Article.id == article_id).first()
        if article:
            article.processing_status = "failed"
            article.processing_error = str(e)
            self.db.commit()

        # Retry
        raise self.retry(exc=e)


@celery_app.task(name="backend.tasks.processing_tasks.save_faiss_index")
def save_faiss_index() -> Dict:
    """
    Save FAISS index to disk (periodic backup).

    Called by:
        - Celery Beat (every hour)
        - After batch processing
    """
    logger.info("Saving FAISS index (periodic backup)...")

    try:
        vector_store = get_vector_store()
        vector_store.save(settings.faiss_index_path)

        stats = vector_store.get_stats()

        logger.info(f"FAISS index saved successfully: {stats}")

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@celery_app.task(name="backend.tasks.processing_tasks.reprocess_failed_articles")
def reprocess_failed_articles() -> Dict:
    """
    Retry processing of failed articles.

    Useful for recovering from temporary errors.
    """
    logger.info("Reprocessing failed articles...")

    try:
        db = SessionLocal()

        # Reset failed articles to pending
        failed_articles = (
            db.query(Article)
            .filter(Article.processing_status == "failed")
            .all()
        )

        count = 0
        for article in failed_articles:
            article.processing_status = "pending"
            article.processing_error = None
            count += 1

        db.commit()
        db.close()

        logger.info(f"Reset {count} failed articles to pending")

        # Trigger processing
        if count > 0:
            process_pending_articles.delay()

        return {
            "status": "success",
            "reset_count": count
        }

    except Exception as e:
        logger.error(f"Error reprocessing failed articles: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    

@celery_app.task(name="backend.tasks.processing_tasks.get_processing_stats")
def get_processing_stats() -> Dict:
    """
    Get statistics about article processing.

    Returns counts of articles in each processing status.
    """
    try:
        db = SessionLocal()

        total = db.query(Article).count()
        pending = db.query(Article).filter(Article.processing_status == "pending").count()
        processing = db.query(Article).filter(Article.processing_status == "processing").count()
        completed = db.query(Article).filter(Article.processing_status == "completed").count()
        failed = db.query(Article).filter(Article.processing_status == "failed").count()

        vector_store = get_vector_store()
        faiss_stats = vector_store.get_stats()

        db.close()

        return {
            "articles": {
                "total": total,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed
            },
            "faiss": faiss_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@celery_app.task(name="backend.tasks.processing_tasks.test_task")
def test_task() -> Dict:
    return {
        "status": "success",
        "message": "test task 0_0"
    }