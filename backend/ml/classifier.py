"""
Topic Classification Module

Loads and serves predictions from the fine-tuned DistilBERT model.
Classifies news articles into 4 categories: World, Sports, Business, Sci/Tech

Model: DistilBERT-base-uncased fine-tuned on AG News dataset
Accuracy: 94.81%
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
from functools import lru_cache
import time

from config.settings import get_settings

logger = logging.getLogger(__name__)

class TopicClassifier:
    """
    Singleton wrapper for the fine-tuned DistilBERT topic classifier.

    Handles:
    - Model loading (lazy, on first use)
    - Device detection (CPU/GPU)
    - Tokenization with proper truncation
    - Batch inference for efficiency
    - Thread-safe operations

    Topics:
    - 0: World
    - 1: Sports
    - 2: Business
    - 3: Sci/Tech
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - only one instance of the classifier exists"""
        if cls._instance is None:
            cls._instance = super(TopicClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the classifier (only runs once due to singleton)"""
        if not TopicClassifier._initialized:
            self.settings = get_settings()
            self.model = None
            self.tokenizer = None
            self.device = None
            self.id2label = None
            self.label2id = None
            TopicClassifier._initialized = True
            logger.info("TopicClassifier instance created (not loaded yet)")

    def load_model(self) -> None:
        """
        Load the model and tokenizer from disk.
        Called automatically on first prediction.

        Raises:
            FileNotFoundError: If model directory doesn't exist
            Exception: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded, skipping")
            return

        logger.info("Loading topic classification model...")
        start_time = time.time()

        try:
            # Get model path from settings (deployment-friendly)
            model_path = Path(self.settings.topic_classifier_path)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Please check TOPIC_CLASSIFIER_PATH in .env file."
                )

            # Detect device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            logger.debug(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Load model
            logger.debug(f"Loading model from {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode (disables dropout)

            # Get label mappings from config
            # Ensure keys are integers for id2label
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            self.label2id = {k: int(v) for k, v in self.model.config.label2id.items()}

            elapsed = time.time() - start_time
            logger.info(f"Model loaded successfully in {elapsed:.2f}s")
            logger.info(f"Topics: {list(self.id2label.values())}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def classify_text(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Dict:
        """
        Classify a single text into one of 4 topics.

        Args:
            text: The article text to classify
            return_all_scores: If True, return scores for all classes

        Returns:
            Dictionary with:
                - topic: The predicted topic name (str)
                - confidence: Confidence score 0-1 (float)
                - topic_id: The topic ID 0-3 (int)
                - all_scores: (optional) Dict of all topic scores

        Example:
            >>> classifier = TopicClassifier()
            >>> result = classifier.classify_text("Tesla announces new electric car")
            >>> print(result)
            {'topic': 'Sci/Tech', 'confidence': 0.95, 'topic_id': 3}
        """
        # Lazy loading - load model on first use
        if self.model is None:
            self.load_model()

        # Handle edge cases
        if not text or not text.strip():
            logger.warning("Empty text provided for classification")
            return {
                "topic": "Unknown",
                "confidence": 0.0,
                "topic_id": -1
            }

        try:
            # Tokenize with truncation (max 512 tokens for DistilBERT)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference (no gradient calculation for speed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get prediction
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
            topic = self.id2label[pred_id]

            result = {
                "topic": topic,
                "confidence": float(confidence),
                "topic_id": int(pred_id)
            }

            # Optionally include probs of all topics
            if return_all_scores:
                all_scores = {
                    self.id2label[i]: float(probs[0][i].item())
                    for i in range(len(self.id2label))
                }
                result["all_scores"] = all_scores

            return result

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify multiple texts efficiently in batches.

        Args:
            texts: List of article texts to classify
            batch_size: Number of texts to process at once

        Returns:
            List of prediction dictionaries, one per input text

        Example:
            >>> classifier = TopicClassifier()
            >>> texts = ["Article 1...", "Article 2...", "Article 3..."]
            >>> results = classifier.classify_batch(texts)
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return []

        results = []

        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # Extract predictions for each text in batch
                for j in range(len(batch)):
                    pred_id = torch.argmax(probs[j], dim=-1).item()
                    confidence = probs[j][pred_id].item()
                    topic = self.id2label[pred_id]

                    results.append({
                        "topic": topic,
                        "confidence": float(confidence),
                        "topic_id": int(pred_id)
                    })

            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                # Add error results for this batch
                for _ in batch:
                    results.append({
                        "topic": "Error",
                        "confidence": 0.0,
                        "topic_id": -1
                    })

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_type": self.model.config.model_type,
            "num_labels": len(self.id2label),
            "topics": list(self.id2label.values()),
            "device": str(self.device),
            "max_length": self.tokenizer.model_max_length
        }


# Global singleton instance
@lru_cache(maxsize=1)
def get_classifier() -> TopicClassifier:
    """
    Get the global TopicClassifier instance (cached).

    This is the recommended way to access the classifier.

    Returns:
        TopicClassifier: The singleton classifier instance

    Example:
        >>> classifier = get_classifier()
        >>> result = classifier.classify_text("Breaking news...")
    """
    return TopicClassifier()


# Convenience function for simple use cases
def classify_text(text: str, return_all_scores: bool = False) -> Dict:
    """
    Convenience function to classify text without managing the classifier instance.

    Args:
        text: The article text to classify
        return_all_scores: If True, return scores for all classes

    Returns:
        Dictionary with prediction results

    Example:
        >>> from backend.ml import classify_text
        >>> result = classify_text("Tesla announces new electric car")
        >>> print(f"Topic: {result['topic']}, Confidence: {result['confidence']:.2f}")
    """
    classifier = get_classifier()
    return classifier.classify_text(text, return_all_scores=return_all_scores)


if __name__ == "__main__":
    # Quick test when running this file directly
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("Testing Topic Classifier")
    print("="*70 + "\n")

    # Test samples for each category
    test_samples = {
        "World": "The United Nations held an emergency meeting today to discuss the ongoing conflict in the Middle East.",
        "Sports": "Manchester United defeated Liverpool 3-1 in a thrilling Premier League match at Old Trafford.",
        "Business": "Apple stock rises 5% after announcing record quarterly earnings and new product lineup.",
        "Sci/Tech": "NASA's James Webb Space Telescope captures stunning images of distant galaxies formed shortly after the Big Bang."
    }

    classifier = get_classifier()

    print("Model Info:")
    print(classifier.get_model_info())
    print("\n")

    for expected_topic, text in test_samples.items():
        result = classifier.classify_text(text, return_all_scores=True)

        print(f"Expected: {expected_topic}")
        print(f"Text: {text[:80]}...")
        print(f"Predicted: {result['topic']} (confidence: {result['confidence']:.3f})")

        if result['all_scores']:
            print("All scores:")
            for topic, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {topic}: {score:.3f}")

        match = "✓" if result['topic'] == expected_topic else "✗"
        print(f"{match}\n")

    print("="*70)
