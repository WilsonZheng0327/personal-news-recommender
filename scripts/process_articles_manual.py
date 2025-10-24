"""
Manual Article Processing Script

Process articles WITHOUT Celery - useful for:
- Testing the pipeline before setting up Celery/Redis
- Debugging processing issues
- One-time bulk processing
- Understanding the flow

This script does the same work as the Celery tasks, but runs synchronously.

Usage:
    python scripts/process_articles_manual.py
    python scripts/process_articles_manual.py --batch-size 100
    python scripts/process_articles_manual.py --article-id 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timezone
import logging

from backend.db.database import SessionLocal
from backend.db.models import Article
from backend.ml.classifier import get_classifier
from backend.ml.embedder import get_embedder
from backend.ml.vector_store import get_vector_store
from config.settings import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_article(article_id: int, db):
    """Process a single article by ID"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing article ID: {article_id}")
    logger.info('='*70)

    # Get article
    article = db.query(Article).filter(Article.id == article_id).first()

    if not article:
        logger.error(f"Article {article_id} not found!")
        return False

    logger.info(f"Title: {article.title}")
    logger.info(f"Source: {article.source}")
    logger.info(f"Current status: {article.processing_status}")

    try:
        # Mark as processing
        article.processing_status = "processing"
        db.commit()

        # Load models
        logger.info("\nLoading ML models...")
        classifier = get_classifier()
        embedder = get_embedder()
        vector_store = get_vector_store()

        # 1. Classify topic
        logger.info("\nStep 1: Classifying topic...")
        classification_text = f"{article.title}\n{article.content[:1000]}"
        result = classifier.classify_text(classification_text, return_all_scores=True)

        article.topic = result["topic"]
        article.topic_confidence = result["confidence"]

        logger.info(f"  Predicted topic: {result['topic']}")
        logger.info(f"  Confidence: {result['confidence']:.3f}")

        if result.get('all_scores'):
            logger.info("  All scores:")
            for topic, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {topic}: {score:.3f}")

        # 2. Generate embedding
        logger.info("\nStep 2: Generating embedding...")
        embedding = embedder.embed_text(article.content)
        logger.info(f"  Embedding shape: {embedding.shape}")
        logger.info(f"  Embedding norm: {(embedding ** 2).sum() ** 0.5:.3f}")

        # 3. Add to FAISS index
        logger.info("\nStep 3: Adding to FAISS index...")
        faiss_position = vector_store.add_vector(embedding, article_id)

        article.embedding_id = faiss_position
        logger.info(f"  FAISS position: {faiss_position}")

        # 4. Update article
        article.processing_status = "completed"
        article.processed_at = datetime.now(timezone.utc)
        article.processing_error = None

        db.commit()

        # 5. Save FAISS index
        settings = get_settings()
        logger.info(f"\nStep 4: Saving FAISS index to {settings.faiss_index_path}...")
        vector_store.save(settings.faiss_index_path)

        logger.info(f"\n{'='*70}")
        logger.info("✅ Article processed successfully!")
        logger.info('='*70)

        return True

    except Exception as e:
        logger.error(f"\n❌ Error processing article: {e}", exc_info=True)

        article.processing_status = "failed"
        article.processing_error = str(e)
        db.commit()

        return False


def process_batch(batch_size: int, db):
    """Process a batch of pending articles"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Batch Processing (max {batch_size} articles)")
    logger.info('='*70)

    # Query pending articles
    logger.info("\nQuerying for pending articles...")
    pending_articles = (
        db.query(Article)
        .filter(Article.processing_status == "pending")
        .limit(batch_size)
        .all()
    )

    if not pending_articles:
        logger.info("✅ No pending articles found!")
        return

    logger.info(f"Found {len(pending_articles)} pending articles\n")

    # Load models once (they're singletons, so this is efficient)
    logger.info("Loading ML models...")
    classifier = get_classifier()
    embedder = get_embedder()
    vector_store = get_vector_store()
    settings = get_settings()

    logger.info(f"  Classifier: {classifier.get_model_info()}")
    logger.info(f"  Embedder: {embedder.get_model_info()}")
    logger.info(f"  Vector store: {vector_store.get_stats()}\n")

    # Process articles
    processed_count = 0
    failed_count = 0

    article_ids = []
    texts_to_embed = []

    logger.info("Processing articles...")
    logger.info("-" * 70)

    for i, article in enumerate(pending_articles, 1):
        try:
            # Mark as processing
            article.processing_status = "processing"
            db.commit()

            logger.info(f"\n[{i}/{len(pending_articles)}] Article {article.id}")
            logger.info(f"  Title: {article.title[:60]}...")

            # 1. Classify
            classification_text = f"{article.title}\n{article.content[:1000]}"
            result = classifier.classify_text(classification_text)

            article.topic = result["topic"]
            article.topic_confidence = result["confidence"]

            logger.info(f"  Topic: {result['topic']} (confidence: {result['confidence']:.3f})")

            # 2. Prepare for embedding
            article_ids.append(article.id)
            texts_to_embed.append(article.content)

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            article.processing_status = "failed"
            article.processing_error = str(e)
            failed_count += 1
            db.commit()

    # 3. Generate embeddings in batch (more efficient)
    if texts_to_embed:
        logger.info(f"\n\nGenerating embeddings for {len(texts_to_embed)} articles...")
        embeddings = embedder.embed_batch(texts_to_embed, show_progress=True)
        logger.info(f"  Embeddings shape: {embeddings.shape}")

        # 4. Add to FAISS
        logger.info(f"\nAdding {len(embeddings)} vectors to FAISS index...")
        faiss_positions = vector_store.add_vectors(embeddings, article_ids)

        # 5. Update articles
        logger.info("\nUpdating articles in database...")
        for article_id, faiss_position in zip(article_ids, faiss_positions):
            article = db.query(Article).filter(Article.id == article_id).first()
            if article:
                article.embedding_id = faiss_position
                article.processing_status = "completed"
                article.processed_at = datetime.now(timezone.utc)
                article.processing_error = None
                processed_count += 1

        db.commit()

        # 6. Save FAISS index
        logger.info(f"\nSaving FAISS index to {settings.faiss_index_path}...")
        vector_store.save(settings.faiss_index_path)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("📊 Processing Summary")
    logger.info('='*70)
    logger.info(f"  Total processed: {processed_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Success rate: {processed_count / len(pending_articles) * 100:.1f}%")

    # Show overall stats
    total = db.query(Article).count()
    pending = db.query(Article).filter(Article.processing_status == "pending").count()
    completed = db.query(Article).filter(Article.processing_status == "completed").count()
    failed = db.query(Article).filter(Article.processing_status == "failed").count()

    logger.info(f"\n📈 Overall Database Stats")
    logger.info(f"  Total articles: {total}")
    logger.info(f"  Pending: {pending}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Failed: {failed}")

    faiss_stats = vector_store.get_stats()
    logger.info(f"\n🔍 FAISS Index Stats")
    for key, value in faiss_stats.items():
        logger.info(f"  {key}: {value}")

    logger.info('='*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Manually process articles")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of articles to process (default: 50)"
    )
    parser.add_argument(
        "--article-id",
        type=int,
        help="Process a specific article by ID"
    )

    args = parser.parse_args()

    # Create database session
    db = SessionLocal()

    try:
        if args.article_id:
            # Process single article
            process_single_article(args.article_id, db)
        else:
            # Process batch
            process_batch(args.batch_size, db)

    finally:
        db.close()


if __name__ == "__main__":
    main()
