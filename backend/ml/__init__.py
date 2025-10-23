"""
Machine Learning module for News Recommender

This module contains ML-related functionality:
- Topic classification (fine-tuned DistilBERT)
- Embedding generation (sentence-transformers)
- Vector search (FAISS)
"""

from .classifier import TopicClassifier, get_classifier, classify_text
from .embedder import ArticleEmbedder, get_embedder, embed_text, embed_batch
from .vector_store import FAISSVectorStore, get_vector_store

__all__ = [
    # Classifier
    "TopicClassifier",
    "get_classifier",
    "classify_text",

    # Embedder
    "ArticleEmbedder",
    "get_embedder",
    "embed_text",
    "embed_batch",

    # Vector Store
    "FAISSVectorStore",
    "get_vector_store",
]