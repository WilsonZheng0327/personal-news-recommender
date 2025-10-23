"""
Test script for Article Embedder

Tests:
1. Model loading and initialization
2. Single text embedding
3. Batch embedding
4. Similarity computation
5. Article embedding (title + content)
6. Performance benchmarking

Usage:
    python scripts/test_embedder.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from backend.ml import get_embedder, embed_text, embed_batch

def test_model_loading():
    """Test that model loads successfully"""
    print("\n" + "="*70)
    print("TEST 1: Model Loading")
    print("="*70)

    embedder = get_embedder()
    info = embedder.get_model_info()

    if info['status'] == 'not_loaded':
        print("[*] Model not loaded yet, triggering load...")
        print("[!] First run will download model (~80MB from HuggingFace)")
        embedder.load_model()
        info = embedder.get_model_info()

    print(f"[+] Model Status: {info['status']}")
    print(f"[+] Model Name: {info.get('model_name', 'N/A')}")
    print(f"[+] Dimension: {info.get('dimension', 'N/A')}")
    print(f"[+] Max Sequence Length: {info.get('max_seq_length', 'N/A')}")


def test_single_embedding():
    """Test embedding a single text"""
    print("\n" + "="*70)
    print("TEST 2: Single Text Embedding")
    print("="*70 + "\n")

    embedder = get_embedder()

    test_texts = [
        "Tesla announces breakthrough in electric vehicle battery technology.",
        "The United Nations discusses climate change policies at global summit.",
        "LeBron James scores 40 points in championship game victory.",
        "Apple stock rises 5% following strong quarterly earnings report.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}:")
        print(f"  Text: {text[:60]}...")

        start_time = time.time()
        embedding = embedder.embed_text(text)
        elapsed = (time.time() - start_time) * 1000

        print(f"  Shape: {embedding.shape}")
        print(f"  Type: {type(embedding)}")
        print(f"  Dtype: {embedding.dtype}")
        print(f"  Norm: {np.linalg.norm(embedding):.3f} (should be ~1.0 if normalized)")
        print(f"  First 5 values: [{', '.join([f'{v:.3f}' for v in embedding[:5]])}]")
        print(f"  Time: {elapsed:.1f}ms")
        print()


def test_batch_embedding():
    """Test batch embedding performance"""
    print("\n" + "="*70)
    print("TEST 3: Batch Embedding")
    print("="*70 + "\n")

    embedder = get_embedder()

    # Create test batch
    texts = [
        "Breaking news from the United Nations headquarters.",
        "The championship game went into overtime.",
        "Stock markets rallied on positive economic data.",
        "Scientists discover new method for quantum computing.",
        "International trade summit concludes with new agreements.",
        "Olympic athlete breaks world record in track event.",
        "Tech companies announce layoffs amid economic uncertainty.",
        "New archaeological discovery sheds light on ancient civilization.",
    ]

    print(f"Embedding {len(texts)} texts in batch...")

    start_time = time.time()
    embeddings = embedder.embed_batch(texts, batch_size=8)
    elapsed = (time.time() - start_time) * 1000

    print(f"[+] Batch embedding completed")
    print(f"    Shape: {embeddings.shape}")
    print(f"    Total time: {elapsed:.1f}ms")
    print(f"    Per text: {elapsed/len(texts):.1f}ms")
    print(f"    Throughput: {len(texts)/(elapsed/1000):.1f} texts/second")


def test_similarity():
    """Test similarity computation"""
    print("\n" + "="*70)
    print("TEST 4: Similarity Computation")
    print("="*70 + "\n")

    embedder = get_embedder()

    # Similar texts (same topic)
    text1 = "Tesla announces new electric vehicle with extended range."
    text2 = "Ford unveils electric car lineup for next year."

    # Different topic
    text3 = "Lakers win NBA championship in thrilling game."

    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)
    emb3 = embedder.embed_text(text3)

    sim_12 = embedder.compute_similarity(emb1, emb2)
    sim_13 = embedder.compute_similarity(emb1, emb3)
    sim_23 = embedder.compute_similarity(emb2, emb3)

    print("Text 1 (EV/Tesla):")
    print(f"  {text1}\n")

    print("Text 2 (EV/Ford):")
    print(f"  {text2}\n")

    print("Text 3 (Sports):")
    print(f"  {text3}\n")

    print("Similarities:")
    print(f"  Text 1 vs Text 2 (similar topics): {sim_12:.3f}")
    print(f"  Text 1 vs Text 3 (different topics): {sim_13:.3f}")
    print(f"  Text 2 vs Text 3 (different topics): {sim_23:.3f}")
    print()

    if sim_12 > sim_13:
        print("[PASS] Similar topics have higher similarity")
    else:
        print("[FAIL] Similar topics should have higher similarity")


def test_batch_similarity():
    """Test batch similarity computation"""
    print("\n" + "="*70)
    print("TEST 5: Batch Similarity Search")
    print("="*70 + "\n")

    embedder = get_embedder()

    # Query article
    query_text = "Artificial intelligence breakthrough in natural language processing"

    # Candidate articles
    candidates = [
        "New AI model achieves human-level language understanding",  # Very similar
        "Machine learning algorithm improves medical diagnosis",      # Related
        "Stock market reaches new highs on tech sector gains",        # Different
        "Football team wins championship after overtime victory",     # Very different
        "Researchers develop advanced neural network architecture",   # Similar
    ]

    print(f"Query: {query_text}\n")
    print("Finding most similar from candidates:\n")

    # Embed all
    query_emb = embedder.embed_text(query_text)
    candidate_embs = embedder.embed_batch(candidates)

    # Compute similarities
    similarities = embedder.compute_similarity_batch(query_emb, candidate_embs)

    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1]  # Descending order

    print("Ranked by similarity:")
    for rank, idx in enumerate(ranked_indices, 1):
        sim = similarities[idx]
        text = candidates[idx]
        print(f"{rank}. [{sim:.3f}] {text}")


def test_article_embedding():
    """Test embedding with title + content"""
    print("\n" + "="*70)
    print("TEST 6: Article Embedding (Title + Content)")
    print("="*70 + "\n")

    embedder = get_embedder()

    # Sample article
    title = "SpaceX Successfully Launches Starship Rocket"
    content = """
    SpaceX achieved a major milestone today with the successful launch of its
    Starship rocket from the company's facility in Texas. The spacecraft reached
    orbital velocity and demonstrated key capabilities needed for future missions
    to the Moon and Mars. CEO Elon Musk called it a "historic day for spaceflight"
    and thanked the engineering team for their efforts. This marks the third attempt
    after previous launches encountered technical difficulties. The rocket will now
    undergo a series of tests before being certified for crewed missions.
    """

    print(f"Title: {title}")
    print(f"Content: {len(content)} characters\n")

    start_time = time.time()
    embedding = embedder.embed_article(title, content)
    elapsed = (time.time() - start_time) * 1000

    print(f"[+] Article embedded successfully")
    print(f"    Shape: {embedding.shape}")
    print(f"    Time: {elapsed:.1f}ms")
    print()

    # Compare with just title
    title_only_emb = embedder.embed_text(title)
    content_only_emb = embedder.embed_text(content)

    sim_full_vs_title = embedder.compute_similarity(embedding, title_only_emb)
    sim_full_vs_content = embedder.compute_similarity(embedding, content_only_emb)

    print("Similarity analysis:")
    print(f"  Full article vs Title only: {sim_full_vs_title:.3f}")
    print(f"  Full article vs Content only: {sim_full_vs_content:.3f}")


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "="*70)
    print("TEST 7: Convenience Functions")
    print("="*70 + "\n")

    # Test embed_text
    print("1. Testing embed_text() function:")
    vector = embed_text("Test article about technology")
    print(f"   [+] embed_text() returned shape {vector.shape}")

    # Test embed_batch
    print("\n2. Testing embed_batch() function:")
    vectors = embed_batch(["Text 1", "Text 2", "Text 3"])
    print(f"   [+] embed_batch() returned shape {vectors.shape}")


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases")
    print("="*70 + "\n")

    embedder = get_embedder()

    # Test 1: Empty string
    print("1. Empty string:")
    emb = embedder.embed_text("")
    print(f"   Shape: {emb.shape}")
    print(f"   All zeros: {np.allclose(emb, 0)}")
    print()

    # Test 2: Very short text
    print("2. Very short text:")
    emb = embedder.embed_text("Hi")
    print(f"   Shape: {emb.shape}")
    print(f"   Norm: {np.linalg.norm(emb):.3f}")
    print()

    # Test 3: Very long text
    print("3. Very long text (should be truncated):")
    long_text = "This is a test. " * 500  # ~7500 chars
    start = time.time()
    emb = embedder.embed_text(long_text)
    elapsed = (time.time() - start) * 1000
    print(f"   Text length: {len(long_text)} chars")
    print(f"   Shape: {emb.shape}")
    print(f"   Time: {elapsed:.1f}ms")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ARTICLE EMBEDDER TEST SUITE")
    print("="*70)

    start_time = time.time()

    try:
        test_model_loading()
        test_single_embedding()
        test_batch_embedding()
        test_similarity()
        test_batch_similarity()
        test_article_embedding()
        test_convenience_functions()
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
