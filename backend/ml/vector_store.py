"""
FAISS Vector Store Module

Facebook AI Similarity Search (FAISS) algorithm

Fast similarity search over article embeddings using Facebook's FAISS library.

Purpose:
- Store embeddings for all articles
- Quick similarity search (find similar articles)
- Build user profiles (find articles similar to user's preferences)
- Content-based recommendations

FAISS Index Types:
- IndexFlatL2: Exact search, good for < 1M vectors (what we use)
- IndexIVFFlat: Approximate search, faster for 1M+ vectors
- IndexIVFPQ: Compressed, for production scale

Performance:
- 10K articles: ~1ms query time
- 100K articles: ~5ms query time
- Memory: ~15MB per 10K articles (384-dim vectors)
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from functools import lru_cache
import pickle

from config.settings import get_settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Singleton FAISS-based vector store for article embeddings.

    Only one instance exists per application to ensure consistency
    and efficient memory usage.

    Handles:
    - Adding embeddings to index
    - Fast similarity search
    - Index persistence (save/load)
    - ID mapping (FAISS index position ↔ article ID)

    Usage:
        store = FAISSVectorStore(dimension=384)
        store.add_vectors(embeddings, ids=[1, 2, 3])
        similar_ids, distances = store.search(query_vector, k=10)

    Recommended:
        store = get_vector_store()  # Use this instead
    """

    _instance = None
    _initialized = False

    def __new__(cls, dimension: int = 384, index_type: str = "flat"):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            cls._instance = super(FAISSVectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize the vector store (only runs once due to singleton).

        Args:
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index
                - "flat": Exact search (IndexFlatL2) - recommended for < 1M vectors
                - "ivf": Approximate search (IndexIVFFlat) - for scaling
        """
        if not FAISSVectorStore._initialized:
            self.dimension = dimension
            self.index_type = index_type

            # FAISS index
            self.index = None

            # ID mapping: FAISS position → Article ID
            # FAISS uses sequential IDs (0, 1, 2, ...), we need to map to actual article IDs
            self.id_map = []  # List where position = FAISS ID, value = Article ID

            # Reverse mapping for fast lookup: Article ID → FAISS position
            self.reverse_id_map = {}

            # Initialize index
            self._initialize_index()

            FAISSVectorStore._initialized = True
            logger.info(f"FAISSVectorStore initialized (dimension={dimension}, type={index_type})")

    def _initialize_index(self) -> None:
        """Initialize the FAISS index based on type."""
        if self.index_type == "flat":
            # Exact search using L2 (Euclidean) distance
            # Good for < 1 million vectors
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.debug("Created IndexFlatL2 (exact search)")

        elif self.index_type == "ivf":
            # Approximate search using Inverted File Index
            # Faster for large datasets, slightly less accurate
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.debug(f"Created IndexIVFFlat (approximate search, nlist={nlist})")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add_vector(
        self,
        embedding: np.ndarray,
        article_id: int
    ) -> int:
        """
        Add a single embedding to the index.

        Args:
            embedding: Embedding vector (384,)
            article_id: Article ID from database

        Returns:
            FAISS index position

        Example:
            >>> store = FAISSVectorStore()
            >>> embedding = embed_text("Article about AI...")
            >>> faiss_id = store.add_vector(embedding, article_id=42)
        """
        # Ensure 2D shape for FAISS (1, 384)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Ensure float32 (FAISS requirement)
        embedding = embedding.astype(np.float32)

        # Add to FAISS index
        self.index.add(embedding)

        # Get FAISS position (sequential)
        faiss_position = len(self.id_map)

        # Store mapping
        self.id_map.append(article_id)
        self.reverse_id_map[article_id] = faiss_position

        logger.debug(f"Added vector for article {article_id} at position {faiss_position}")

        return faiss_position

    def add_vectors(
        self,
        embeddings: np.ndarray,
        article_ids: List[int]
    ) -> List[int]:
        """
        Add multiple embeddings to the index in batch.

        Args:
            embeddings: Array of embeddings (N, 384)
            article_ids: List of article IDs (length N)

        Returns:
            List of FAISS positions

        Example:
            >>> embeddings = embed_batch(["Article 1", "Article 2", "Article 3"])
            >>> positions = store.add_vectors(embeddings, article_ids=[10, 20, 30])
        """
        if len(embeddings) != len(article_ids):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of IDs ({len(article_ids)})"
            )

        # Ensure 2D shape
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Ensure float32
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS
        self.index.add(embeddings)

        # Store mappings
        start_position = len(self.id_map)
        positions = []

        for i, article_id in enumerate(article_ids):
            faiss_position = start_position + i
            self.id_map.append(article_id)
            self.reverse_id_map[article_id] = faiss_position
            positions.append(faiss_position)

        logger.info(f"Added {len(embeddings)} vectors to index")

        return positions

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        return_distances: bool = True
    ) -> Tuple[List[int], Optional[List[float]]]:
        """
        Search for k most similar vectors.

        Args:
            query_vector: Query embedding (384,)
            k: Number of results to return
            return_distances: If True, return distances along with IDs

        Returns:
            Tuple of (article_ids, distances) or just article_ids

        Example:
            >>> query = embed_text("User's favorite article")
            >>> article_ids, distances = store.search(query, k=10)
            >>> print(f"Top 10 similar articles: {article_ids}")
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, cannot search")
            return ([], []) if return_distances else []

        # Ensure 2D shape (1, 384)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure float32
        query_vector = query_vector.astype(np.float32)

        # Limit k to available vectors
        k = min(k, self.index.ntotal)

        # Search FAISS index
        # Returns: distances (lower = more similar), indices (FAISS positions)
        distances, indices = self.index.search(query_vector, k)

        # Convert FAISS indices to article IDs
        article_ids = [self.id_map[idx] for idx in indices[0]]

        # Convert to Python floats
        distances_list = distances[0].tolist()

        logger.debug(f"Search found {len(article_ids)} results")

        if return_distances:
            return article_ids, distances_list
        else:
            return article_ids

    def search_by_article_id(
        self,
        article_id: int,
        k: int = 10,
        exclude_self: bool = True
    ) -> Tuple[List[int], List[float]]:
        """
        Find articles similar to a given article (by its ID).

        Args:
            article_id: ID of the article to find similar articles for
            k: Number of results
            exclude_self: If True, exclude the query article from results

        Returns:
            Tuple of (article_ids, distances)

        Example:
            >>> # User likes article 42, find similar ones
            >>> similar_ids, scores = store.search_by_article_id(42, k=10)
        """
        # Get FAISS position for this article
        if article_id not in self.reverse_id_map:
            raise ValueError(f"Article {article_id} not found in index")

        faiss_position = self.reverse_id_map[article_id]

        # Get the embedding from FAISS
        query_vector = self.index.reconstruct(int(faiss_position))

        # Search (get k+1 if excluding self)
        search_k = k + 1 if exclude_self else k
        article_ids, distances = self.search(query_vector, k=search_k)

        # Remove self if needed
        if exclude_self and article_ids and article_ids[0] == article_id:
            article_ids = article_ids[1:]
            distances = distances[1:]
        elif exclude_self and article_id in article_ids:
            idx = article_ids.index(article_id)
            article_ids.pop(idx)
            distances.pop(idx)

        # Limit to k
        return article_ids[:k], distances[:k]

    def remove_vector(self, article_id: int) -> bool:
        """
        Remove a vector from the index.

        Note: FAISS doesn't support efficient deletion, so this marks the ID as invalid.
        For production, consider rebuilding the index periodically.

        Args:
            article_id: Article ID to remove

        Returns:
            True if removed, False if not found
        """
        if article_id not in self.reverse_id_map:
            logger.warning(f"Article {article_id} not found in index")
            return False

        # Mark as removed (set to -1)
        faiss_position = self.reverse_id_map[article_id]
        self.id_map[faiss_position] = -1
        del self.reverse_id_map[article_id]

        logger.debug(f"Marked article {article_id} as removed")
        return True

    def save(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            index_path: Path to save FAISS index (.bin)
            metadata_path: Path to save metadata (.pkl) - if None, auto-generated

        Example:
            >>> store.save("data/faiss_index.bin")
        """
        # Create directory if needed
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata (ID mappings)
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.pkl'))

        metadata = {
            'id_map': self.id_map,
            'reverse_id_map': self.reverse_id_map,
            'dimension': self.dimension,
            'index_type': self.index_type,
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load the FAISS index and metadata from disk.

        Args:
            index_path: Path to FAISS index (.bin)
            metadata_path: Path to metadata (.pkl) - if None, auto-detected

        Example:
            >>> store = FAISSVectorStore()
            >>> store.load("data/faiss_index.bin")
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")

        # Load metadata
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.pkl'))

        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.id_map = metadata['id_map']
        self.reverse_id_map = metadata['reverse_id_map']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']

        logger.info(f"Loaded metadata from {metadata_path}")
        logger.info(f"Index contains {len(self.id_map)} vectors")

    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'id_map_size': len(self.id_map),
            'is_trained': self.index.is_trained if self.index else False,
        }

    def clear(self) -> None:
        """Clear the index and reset mappings."""
        self._initialize_index()
        self.id_map = []
        self.reverse_id_map = {}
        logger.info("Index cleared")


# Global singleton instance
@lru_cache(maxsize=1)
def get_vector_store() -> FAISSVectorStore:
    """
    Get the global FAISSVectorStore instance (cached).

    Returns:
        FAISSVectorStore: The singleton vector store instance

    Example:
        >>> from backend.ml import get_vector_store
        >>> store = get_vector_store()
        >>> store.add_vector(embedding, article_id=42)
    """
    settings = get_settings()
    store = FAISSVectorStore(dimension=384, index_type="flat")

    # Try to load existing index
    index_path = Path(settings.faiss_index_path)
    if index_path.exists():
        try:
            store.load(str(index_path))
            logger.info("Loaded existing FAISS index")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            logger.info("Starting with empty index")

    return store


if __name__ == "__main__":
    # Quick test when running this file directly
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("Testing FAISS Vector Store")
    print("="*70 + "\n")

    # Create store
    store = FAISSVectorStore(dimension=384)
    print(f"Created vector store: {store.get_stats()}\n")

    # Create some test vectors
    np.random.seed(42)
    test_vectors = np.random.randn(10, 384).astype(np.float32)
    # Normalize (like real embeddings)
    test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=1, keepdims=True)

    test_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Test 1: Add vectors
    print("Test 1: Adding vectors")
    positions = store.add_vectors(test_vectors, test_ids)
    print(f"  Added {len(positions)} vectors")
    print(f"  Stats: {store.get_stats()}\n")

    # Test 2: Search
    print("Test 2: Searching")
    query = test_vectors[0]  # Use first vector as query
    similar_ids, distances = store.search(query, k=5)
    print(f"  Query: Article {test_ids[0]}")
    print(f"  Top 5 similar:")
    for i, (aid, dist) in enumerate(zip(similar_ids, distances), 1):
        print(f"    {i}. Article {aid} (distance: {dist:.4f})")
    print()

    # Test 3: Search by article ID
    print("Test 3: Search by article ID")
    similar_ids, distances = store.search_by_article_id(test_ids[0], k=5)
    print(f"  Similar to Article {test_ids[0]}:")
    for i, (aid, dist) in enumerate(zip(similar_ids, distances), 1):
        print(f"    {i}. Article {aid} (distance: {dist:.4f})")
    print()

    # Test 4: Save and load
    print("Test 4: Save and load")
    test_path = "test_faiss_index.bin"
    store.save(test_path)
    print(f"  Saved to {test_path}")

    new_store = FAISSVectorStore(dimension=384)
    new_store.load(test_path)
    print(f"  Loaded from {test_path}")
    print(f"  Stats: {new_store.get_stats()}")

    # Clean up
    import os
    os.remove(test_path)
    os.remove(test_path.replace('.bin', '.pkl'))
    print(f"  Cleaned up test files\n")

    print("="*70)
    print("All tests completed!")
    print("="*70 + "\n")