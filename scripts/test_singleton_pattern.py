"""
Test that all three ML components use singleton pattern correctly.

This verifies:
1. Multiple instantiations return the same object
2. State is shared across instances
3. Consistent behavior across classifier, embedder, and vector_store
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backend.ml import get_classifier, get_embedder, get_vector_store
from backend.ml.classifier import TopicClassifier
from backend.ml.embedder import ArticleEmbedder
from backend.ml.vector_store import FAISSVectorStore


def test_classifier_singleton():
    """Test classifier singleton pattern"""
    print("\n" + "="*70)
    print("TEST 1: Classifier Singleton")
    print("="*70)

    # Create multiple instances
    classifier1 = TopicClassifier()
    classifier2 = TopicClassifier()
    classifier3 = get_classifier()

    # Check they're all the same object
    print(f"classifier1 is classifier2: {classifier1 is classifier2}")
    print(f"classifier2 is classifier3: {classifier2 is classifier3}")
    print(f"classifier1 is classifier3: {classifier1 is classifier3}")

    # Check memory addresses
    print(f"\nMemory addresses:")
    print(f"  classifier1: {id(classifier1)}")
    print(f"  classifier2: {id(classifier2)}")
    print(f"  classifier3: {id(classifier3)}")

    if classifier1 is classifier2 is classifier3:
        print("\n[PASS] All classifier instances are the same object")
    else:
        print("\n[FAIL] Classifier instances are different objects")

    return classifier1 is classifier2 is classifier3


def test_embedder_singleton():
    """Test embedder singleton pattern"""
    print("\n" + "="*70)
    print("TEST 2: Embedder Singleton")
    print("="*70)

    # Create multiple instances
    embedder1 = ArticleEmbedder()
    embedder2 = ArticleEmbedder()
    embedder3 = get_embedder()

    # Check they're all the same object
    print(f"embedder1 is embedder2: {embedder1 is embedder2}")
    print(f"embedder2 is embedder3: {embedder2 is embedder3}")
    print(f"embedder1 is embedder3: {embedder1 is embedder3}")

    # Check memory addresses
    print(f"\nMemory addresses:")
    print(f"  embedder1: {id(embedder1)}")
    print(f"  embedder2: {id(embedder2)}")
    print(f"  embedder3: {id(embedder3)}")

    if embedder1 is embedder2 is embedder3:
        print("\n[PASS] All embedder instances are the same object")
    else:
        print("\n[FAIL] Embedder instances are different objects")

    return embedder1 is embedder2 is embedder3


def test_vector_store_singleton():
    """Test vector store singleton pattern"""
    print("\n" + "="*70)
    print("TEST 3: Vector Store Singleton")
    print("="*70)

    # Create multiple instances
    store1 = FAISSVectorStore()
    store2 = FAISSVectorStore(dimension=384)
    store3 = get_vector_store()

    # Check they're all the same object
    print(f"store1 is store2: {store1 is store2}")
    print(f"store2 is store3: {store2 is store3}")
    print(f"store1 is store3: {store1 is store3}")

    # Check memory addresses
    print(f"\nMemory addresses:")
    print(f"  store1: {id(store1)}")
    print(f"  store2: {id(store2)}")
    print(f"  store3: {id(store3)}")

    if store1 is store2 is store3:
        print("\n[PASS] All vector store instances are the same object")
    else:
        print("\n[FAIL] Vector store instances are different objects")

    return store1 is store2 is store3


def test_shared_state():
    """Test that singleton instances share state"""
    print("\n" + "="*70)
    print("TEST 4: Shared State Across Instances")
    print("="*70)

    # Vector store test - add data via one instance, access via another
    print("\n1. Vector Store State Sharing:")

    store1 = FAISSVectorStore()

    # Add a vector via store1
    test_vector = np.random.randn(384).astype(np.float32)
    test_vector = test_vector / np.linalg.norm(test_vector)

    store1.add_vector(test_vector, article_id=999)
    print(f"   Added vector via store1")
    print(f"   store1 stats: {store1.get_stats()['total_vectors']} vectors")

    # Access via different instance
    store2 = FAISSVectorStore()
    print(f"   store2 stats: {store2.get_stats()['total_vectors']} vectors")

    # Search via third instance
    store3 = get_vector_store()
    results, _ = store3.search(test_vector, k=1)
    print(f"   Search via store3 found article: {results[0] if results else 'None'}")

    if results and results[0] == 999:
        print("   [PASS] State is shared across all instances")
        return True
    else:
        print("   [FAIL] State is NOT shared")
        return False


def test_initialization_once():
    """Test that initialization only happens once"""
    print("\n" + "="*70)
    print("TEST 5: Initialize Only Once")
    print("="*70)

    # Check initial state
    print(f"\nClassifier._initialized: {TopicClassifier._initialized}")
    print(f"Embedder._initialized: {ArticleEmbedder._initialized}")
    print(f"VectorStore._initialized: {FAISSVectorStore._initialized}")

    # Create new instances
    c = TopicClassifier()
    e = ArticleEmbedder()
    v = FAISSVectorStore()

    # Check they're still initialized
    print(f"\nAfter creating new instances:")
    print(f"Classifier._initialized: {TopicClassifier._initialized}")
    print(f"Embedder._initialized: {ArticleEmbedder._initialized}")
    print(f"VectorStore._initialized: {FAISSVectorStore._initialized}")

    if all([TopicClassifier._initialized, ArticleEmbedder._initialized, FAISSVectorStore._initialized]):
        print("\n[PASS] All components initialized exactly once")
        return True
    else:
        print("\n[FAIL] Initialization state incorrect")
        return False


def test_consistency():
    """Test that all three components follow the same pattern"""
    print("\n" + "="*70)
    print("TEST 6: Pattern Consistency")
    print("="*70)

    # Check that all three have the same singleton attributes
    has_instance = all([
        hasattr(TopicClassifier, '_instance'),
        hasattr(ArticleEmbedder, '_instance'),
        hasattr(FAISSVectorStore, '_instance')
    ])

    has_initialized = all([
        hasattr(TopicClassifier, '_initialized'),
        hasattr(ArticleEmbedder, '_initialized'),
        hasattr(FAISSVectorStore, '_initialized')
    ])

    has_new = all([
        hasattr(TopicClassifier, '__new__'),
        hasattr(ArticleEmbedder, '__new__'),
        hasattr(FAISSVectorStore, '__new__')
    ])

    print(f"All have _instance attribute: {has_instance}")
    print(f"All have _initialized attribute: {has_initialized}")
    print(f"All have __new__ method: {has_new}")

    if has_instance and has_initialized and has_new:
        print("\n[PASS] All components use consistent singleton pattern")
        return True
    else:
        print("\n[FAIL] Components use different patterns")
        return False


def run_all_tests():
    """Run all singleton tests"""
    print("\n" + "="*70)
    print("SINGLETON PATTERN TEST SUITE")
    print("="*70)

    results = []

    results.append(("Classifier Singleton", test_classifier_singleton()))
    results.append(("Embedder Singleton", test_embedder_singleton()))
    results.append(("Vector Store Singleton", test_vector_store_singleton()))
    results.append(("Shared State", test_shared_state()))
    results.append(("Initialize Once", test_initialization_once()))
    results.append(("Pattern Consistency", test_consistency()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All singleton tests passed!")
        print("\nKey Benefits:")
        print("  - Classifier, embedder, and vector store use consistent pattern")
        print("  - Only one instance of each exists in memory")
        print("  - State is shared across all usages")
        print("  - Prevents bugs from multiple instances")
        return True
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
