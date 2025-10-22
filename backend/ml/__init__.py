"""
Machine Learning module for News Recommender

This module contains ML-related functionality:
- Topic classification (fine-tuned DistilBERT)
- Embedding generation (sentence-transformers)
- Vector search (FAISS)
"""

from .classifier import TopicClassifier, get_classifier, classify_text

__all__ = [
    "TopicClassifier",
    "get_classifier",
    "classify_text"
]