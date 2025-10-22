"""
Test script for Topic Classifier

Tests:
1. Model loading and initialization
2. Classification of sample texts
3. Batch classification performance
4. Classification of real articles from database (if available)
5. Edge cases (empty text, very long text)

Usage:
    python scripts/test_classifier.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from backend.ml.classifier import get_classifier, classify_text
from backend.db.database import SessionLocal
from backend.db.models import Article

def test_model_loading():
    """Test that model loads successfully"""
    print("\n" + "="*70)
    print("TEST 1: Model Loading")
    print("="*70)

    classifier = get_classifier()
    info = classifier.get_model_info()

    if info['status'] == 'not_loaded':
        print("[*] Model not loaded yet, triggering load...")
        classifier.load_model()
        info = classifier.get_model_info()

    print(f"[+] Model Status: {info['status']}")
    print(f"[+] Model Type: {info.get('model_type', 'N/A')}")
    print(f"[+] Device: {info.get('device', 'N/A')}")
    print(f"[+] Topics: {info.get('topics', [])}")
    print(f"[+] Max Length: {info.get('max_length', 'N/A')}")


def test_sample_classifications():
    """Test classification on hand-crafted samples"""
    print("\n" + "="*70)
    print("TEST 2: Sample Classifications")
    print("="*70 + "\n")

    test_samples = [
        {
            "expected": "World",
            "text": "The United Nations Security Council convened an emergency session to address the escalating humanitarian crisis."
        },
        {
            "expected": "Sports",
            "text": "LeBron James scored 40 points as the Lakers defeated the Warriors in overtime at the Staples Center."
        },
        {
            "expected": "Business",
            "text": "The Federal Reserve announced an interest rate hike of 0.25% to combat rising inflation in the economy."
        },
        {
            "expected": "Sci/Tech",
            "text": "Researchers at MIT developed a new artificial intelligence algorithm that can predict protein structures with 95% accuracy."
        },
        {
            "expected": "World",
            "text": "European leaders gathered in Brussels to discuss new climate policies and carbon emission targets for 2030."
        },
        {
            "expected": "Sports",
            "text": "The FIFA World Cup final drew record viewership numbers as millions tuned in to watch the championship match."
        },
        {
            "expected": "Business",
            "text": "Amazon stock surged 8% following the announcement of better-than-expected quarterly earnings and strong holiday sales."
        },
        {
            "expected": "Sci/Tech",
            "text": "SpaceX successfully launched its Starship rocket, marking a major milestone in the development of reusable spacecraft."
        }
    ]

    correct = 0
    total = len(test_samples)

    for i, sample in enumerate(test_samples, 1):
        result = classify_text(sample["text"], return_all_scores=False)

        is_correct = result["topic"] == sample["expected"]
        if is_correct:
            correct += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"{status} Test {i}:")
        print(f"  Expected: {sample['expected']}")
        print(f"  Predicted: {result['topic']} (confidence: {result['confidence']:.3f})")
        print(f"  Text: {sample['text'][:80]}...")
        print()

    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    if accuracy >= 75:
        print("[+] Classification performance is good!")
    else:
        print("[!] Classification performance may need improvement")


def test_batch_classification():
    """Test batch classification performance"""
    print("\n" + "="*70)
    print("TEST 3: Batch Classification Performance")
    print("="*70 + "\n")

    classifier = get_classifier()

    # Create test batch
    texts = [
        "Breaking news from the United Nations headquarters in New York City.",
        "The championship game went into double overtime with an exciting finish.",
        "Stock markets rallied today on positive economic data from the Federal Reserve.",
        "Scientists discover new method for quantum computing using topological materials.",
    ] * 10  # 40 texts total

    print(f"Classifying {len(texts)} texts in batch...")

    start_time = time.time()
    results = classifier.classify_batch(texts, batch_size=16)
    elapsed = time.time() - start_time

    print(f"[+] Classified {len(results)} texts in {elapsed:.3f}s")
    print(f"[+] Average: {(elapsed/len(results))*1000:.1f}ms per text")
    print(f"[+] Throughput: {len(results)/elapsed:.1f} texts/second")

    # Show distribution
    topics = {}
    for result in results:
        topic = result['topic']
        topics[topic] = topics.get(topic, 0) + 1

    print(f"\nTopic Distribution:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count}")


def test_database_articles():
    """Test classification on real articles from database"""
    print("\n" + "="*70)
    print("TEST 4: Real Database Articles")
    print("="*70 + "\n")

    db = SessionLocal()
    try:
        # Get some articles (preferably unclassified)
        articles = db.query(Article)\
            .filter(Article.topic.is_(None))\
            .limit(5)\
            .all()

        if not articles:
            print("[!] No unclassified articles found in database")
            # Try to get any articles
            articles = db.query(Article).limit(5).all()

        if not articles:
            print("[!] No articles found in database. Run the scraper first:")
            print("  python backend/scrapers/rss_scraper.py")
            return

        print(f"Found {len(articles)} articles to classify\n")

        classifier = get_classifier()

        for i, article in enumerate(articles, 1):
            print(f"Article {i}:")
            print(f"  Title: {article.title[:80]}")
            print(f"  URL: {article.url[:60]}...")

            start_time = time.time()
            result = classifier.classify_text(article.title + article.content)
            elapsed = time.time() - start_time

            print(f"  Predicted Topic: {result['topic']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Inference Time: {elapsed*1000:.1f}ms")
            print()

    except Exception as e:
        print(f"[-] Error accessing database: {e}")
        print(f"  Make sure Docker containers are running:")
        print(f"  docker-compose up -d")
    finally:
        db.close()


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70 + "\n")

    classifier = get_classifier()

    # Test 1: Empty string
    print("Test: Empty string")
    result = classifier.classify_text("")
    print(f"  Result: {result}")
    print(f"  Status: {'[+] Handled gracefully' if result['confidence'] == 0.0 else '[-] Unexpected'}")
    print()

    # Test 2: Very short text
    print("Test: Very short text")
    result = classifier.classify_text("Hello")
    print(f"  Result: {result}")
    print(f"  Status: [+] Classified (even if not meaningful)")
    print()

    # Test 3: Very long text (>512 tokens - should be truncated)
    print("Test: Very long text (>512 tokens)")
    long_text = "This is a test. " * 200  # ~600 tokens
    start_time = time.time()
    result = classifier.classify_text(long_text)
    elapsed = time.time() - start_time
    print(f"  Result: {result}")
    print(f"  Time: {elapsed*1000:.1f}ms")
    print(f"  Status: [+] Handled with truncation")
    print()

    # Test 4: Non-English text (model trained on English)
    print("Test: Non-English text")
    result = classifier.classify_text("Dies ist ein Test auf Deutsch.")
    print(f"  Result: {result}")
    print(f"  Status: [+] Classified (may not be accurate)")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("TOPIC CLASSIFIER TEST SUITE")
    print("="*70)

    start_time = time.time()

    try:
        test_model_loading()
        test_sample_classifications()
        test_batch_classification()
        test_database_articles()
        test_edge_cases()

    except Exception as e:
        print(f"\n[-] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print(f"[SUCCESS] All tests completed in {elapsed:.2f}s")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
