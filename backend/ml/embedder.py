"""
Article Embedding Module

Generates semantic embeddings for news articles using sentence-transformers.
Embeddings are used for:
- Content-based recommendations (find similar articles)
- User profile building (aggregate embeddings of liked articles)
- Semantic search (query matching)

Model: all-MiniLM-L6-v2
- Output dimension: 384
- Optimized for semantic similarity tasks
- Fast inference: ~10-20ms per article
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from functools import lru_cache
import time

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ArticleEmbedder:
    """
    Singleton wrapper for sentence-transformers embedding model.

    Handles:
    - Model loading (lazy, on first use)
    - Text encoding to dense vectors
    - Batch processing for efficiency
    - Normalization for cosine similarity

    Output:
    - 384-dimensional normalized vectors
    - Ready for FAISS similarity search
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            # parent class of this is just a Python Object
            cls._instance = super(ArticleEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the embedder (only runs once due to singleton)"""
        if not ArticleEmbedder._initialized:
            self.settings = get_settings()
            self.model = None
            self.model_name = None
            self.dimension = None
            ArticleEmbedder._initialized = True
            logger.info("ArticleEmbedder instance created (not loaded yet)")

    def load_model(self) -> None:
        """
        Load the sentence-transformers model.
        Called automatically on first encoding.

        Raises:
            Exception: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded, skipping")
            return

        logger.info("Loading sentence-transformers model...")
        start_time = time.time()

        try:
            # Get model name from settings (deployment-friendly)
            self.model_name = self.settings.embedding_model_name
            logger.info(f"Model: {self.model_name}")

            # Load model
            # First run will download from HuggingFace (~80MB)
            # Subsequent runs load from cache (~/.cache/torch/sentence_transformers/)
            self.model = SentenceTransformer(self.model_name)

            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()

            elapsed = time.time() - start_time
            logger.info(f"Model loaded successfully in {elapsed:.2f}s")
            logger.info(f"Embedding dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(
        self,
        text: str,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed (article content, title, etc.)
            normalize: If True, normalize to unit length (better for cosine similarity)
            convert_to_numpy: If True, return numpy array; otherwise return tensor

        Returns:
            Embedding vector of shape (384,)

        Example:
            >>> embedder = ArticleEmbedder()
            >>> vector = embedder.embed_text("Tesla announces new electric car")
            >>> vector.shape
            (384,)
        """
        # Lazy loading - load model on first use
        if self.model is None:
            self.load_model()

        # Handle edge cases
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=False
            )

            return embedding

        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            raise

    def embed_article(
        self,
        title: str,
        content: str,
        title_weight: float = 0.3
    ) -> np.ndarray:
        """
        Generate embedding for an article by combining title and content.

        Strategy: Weighted average of title and content embeddings.
        Rationale: Title often contains key concepts, but content has details.

        Args:
            title: Article title
            content: Article content
            title_weight: Weight for title embedding (0-1)

        Returns:
            Combined embedding vector (384,)

        Example:
            >>> embedder = ArticleEmbedder()
            >>> embedding = embedder.embed_article(
            ...     title="Tesla announces new EV",
            ...     content="Tesla Inc. today unveiled..."
            ... )
        """
        if self.model is None:
            self.load_model()

        # Simple approach: concatenate title and content
        # Separator helps model distinguish sections
        combined_text = f"{title}\n\n{content}"

        # Truncate if too long (model has max length ~512 tokens)
        # For all-MiniLM-L6-v2, max_seq_length is 256 word pieces
        # Roughly ~200 words, so truncate content if needed
        max_chars = 2000  # Rough estimate

        if len(combined_text) > max_chars:
            # Keep full title, truncate content
            content_limit = max_chars - len(title) - 10
            if content_limit > 0:
                combined_text = f"{title}\n\n{content[:content_limit]}..."
            else:
                combined_text = title

        return self.embed_text(combined_text)

        # Alternative approach (more complex but potentially better):
        # title_emb = self.embed_text(title)
        # content_emb = self.embed_text(content)
        # combined = title_weight * title_emb + (1 - title_weight) * content_emb
        # # Normalize
        # combined = combined / np.linalg.norm(combined)
        # return combined

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            normalize: If True, normalize vectors to unit length
            show_progress: If True, show progress bar

        Returns:
            Array of embeddings with shape (N, 384)

        Example:
            >>> embedder = ArticleEmbedder()
            >>> texts = ["Article 1...", "Article 2...", "Article 3..."]
            >>> embeddings = embedder.embed_batch(texts)
            >>> embeddings.shape
            (3, 384)
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            # Batch encoding is much faster than individual encodes
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error during batch embedding: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector (384,)
            embedding2: Second embedding vector (384,)

        Returns:
            Similarity score between -1 and 1 (higher = more similar)

        Note:
            If embeddings are normalized, this is just a dot product.

        Example:
            >>> vec1 = embedder.embed_text("Tesla launches new EV")
            >>> vec2 = embedder.embed_text("Ford announces electric vehicle")
            >>> similarity = embedder.compute_similarity(vec1, vec2)
            >>> print(f"Similarity: {similarity:.3f}")
            Similarity: 0.847
        """
        # If normalized, cosine similarity = dot product
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def compute_similarity_batch(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between one query and multiple candidates.

        Args:
            query_embedding: Query vector of shape (384,)
            candidate_embeddings: Candidate vectors of shape (N, 384)

        Returns:
            Similarity scores of shape (N,)

        Example:
            >>> query = embedder.embed_text("User's favorite article")
            >>> candidates = embedder.embed_batch(["Article 1", "Article 2", "Article 3"])
            >>> scores = embedder.compute_similarity_batch(query, candidates)
            >>> best_match_idx = np.argmax(scores)
        """
        # Matrix multiplication: (384,) @ (N, 384).T = (N,)
        similarities = np.dot(candidate_embeddings, query_embedding)
        return similarities

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {
                "status": "not_loaded",
                "model_name": self.settings.embedding_model_name
            }

        return {
            "status": "loaded",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length
        }


# Global singleton instance
@lru_cache(maxsize=1)
def get_embedder() -> ArticleEmbedder:
    """
    Get the global ArticleEmbedder instance (cached).

    This is the recommended way to access the embedder.

    Returns:
        ArticleEmbedder: The singleton embedder instance

    Example:
        >>> from backend.ml import get_embedder
        >>> embedder = get_embedder()
        >>> vector = embedder.embed_text("Article text...")
    """
    return ArticleEmbedder()


# Convenience functions
def embed_text(text: str) -> np.ndarray:
    """
    Convenience function to embed text without managing embedder instance.

    Args:
        text: Text to embed

    Returns:
        Embedding vector (384,)

    Example:
        >>> from backend.ml import embed_text
        >>> vector = embed_text("Tesla announces new electric car")
    """
    embedder = get_embedder()
    return embedder.embed_text(text)


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Convenience function to embed multiple texts.

    Args:
        texts: List of texts to embed

    Returns:
        Embedding matrix (N, 384)

    Example:
        >>> from backend.ml import embed_batch
        >>> vectors = embed_batch(["Text 1", "Text 2", "Text 3"])
    """
    embedder = get_embedder()
    return embedder.embed_batch(texts)

