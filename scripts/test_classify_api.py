"""
Test script for classification API endpoints

Tests:
1. POST /api/classify/text - Classify raw text
2. POST /api/classify/article/{id} - Classify article by ID
3. POST /api/classify/batch - Batch classification
4. GET /api/classify/model-info - Model information

Usage:
    1. Start the API server:
       uvicorn backend.api.main:app --reload

    2. Run this test script:
       python scripts/test_classify_api.py
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def test_model_info():
    """Test GET /api/classify/model-info"""
    print_section("TEST 1: Model Info")

    url = f"{API_BASE_URL}/api/classify/model-info"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print("[+] Model info retrieved successfully")
            print(f"Status: {data.get('status')}")
            print(f"Model Type: {data.get('model_type')}")
            print(f"Topics: {data.get('topics')}")
            print(f"Device: {data.get('device')}")
            return True
        else:
            print(f"[-] Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("[-] Connection failed. Is the API server running?")
        print("    Start it with: uvicorn backend.api.main:app --reload")
        return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def test_classify_text():
    """Test POST /api/classify/text"""
    print_section("TEST 2: Classify Raw Text")

    url = f"{API_BASE_URL}/api/classify/text"

    test_cases = [
        {
            "text": "The United Nations held an emergency meeting to discuss climate change policies.",
            "expected": "World"
        },
        {
            "text": "LeBron James scored 40 points in the championship game last night.",
            "expected": "Sports"
        },
        {
            "text": "Apple stock rises 5% after announcing record quarterly earnings.",
            "expected": "Business"
        },
        {
            "text": "NASA's James Webb Space Telescope captures stunning images of distant galaxies.",
            "expected": "Sci/Tech"
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test['text'][:60]}...")
        print(f"Expected: {test['expected']}")

        try:
            response = requests.post(
                url,
                json={"text": test["text"], "return_all_scores": False}
            )

            if response.status_code == 200:
                data = response.json()
                predicted = data.get('topic')
                confidence = data.get('confidence')

                passed = predicted == test['expected']
                status = "[PASS]" if passed else "[FAIL]"

                print(f"{status} Predicted: {predicted} (confidence: {confidence:.3f})")

                if not passed:
                    all_passed = False
            else:
                print(f"[-] Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                all_passed = False

        except Exception as e:
            print(f"[-] Error: {e}")
            all_passed = False

    return all_passed


def test_classify_text_with_all_scores():
    """Test POST /api/classify/text with all_scores=True"""
    print_section("TEST 3: Classify with All Scores")

    url = f"{API_BASE_URL}/api/classify/text"

    text = "Tesla announces breakthrough in battery technology for electric vehicles."

    print(f"Text: {text}")

    try:
        response = requests.post(
            url,
            json={"text": text, "return_all_scores": True}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[+] Classification successful")
            print(f"Predicted Topic: {data.get('topic')}")
            print(f"Confidence: {data.get('confidence'):.3f}")

            if 'all_scores' in data and data['all_scores']:
                print("\nAll Topic Scores:")
                for topic, score in sorted(
                    data['all_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    print(f"  {topic}: {score:.3f}")
                return True
            else:
                print("[-] all_scores not returned")
                return False
        else:
            print(f"[-] Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def test_batch_classify():
    """Test POST /api/classify/batch"""
    print_section("TEST 4: Batch Classification")

    url = f"{API_BASE_URL}/api/classify/batch"

    texts = [
        "Breaking news from the White House today.",
        "The Lakers won the championship game.",
        "Stock markets hit all-time highs.",
        "Scientists discover new exoplanet.",
        "International summit on global trade.",
        "Olympic athlete breaks world record."
    ]

    print(f"Classifying {len(texts)} texts in batch...")

    try:
        response = requests.post(
            url,
            json={"texts": texts}
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            total = data.get('total')
            time_ms = data.get('processing_time_ms')

            print(f"\n[+] Batch classification successful")
            print(f"Total: {total}")
            print(f"Processing Time: {time_ms:.2f}ms")
            print(f"Average: {time_ms/total:.2f}ms per text")

            print("\nResults:")
            for i, (text, result) in enumerate(zip(texts, results), 1):
                print(f"{i}. {text[:50]:50s} -> {result['topic']:12s} ({result['confidence']:.3f})")

            return True
        else:
            print(f"[-] Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def test_classify_article():
    """Test POST /api/classify/article/{id}"""
    print_section("TEST 5: Classify Article by ID")

    # First, get an article ID from the database
    articles_url = f"{API_BASE_URL}/api/articles/recent?limit=1"

    try:
        response = requests.get(articles_url)

        if response.status_code != 200 or not response.json():
            print("[!] No articles found in database")
            print("    Run the scraper first: python backend/scrapers/rss_scraper.py")
            return None  # Skip this test

        article = response.json()[0]
        article_id = article['id']
        article_title = article['title']

        print(f"Testing with Article ID: {article_id}")
        print(f"Title: {article_title[:60]}...")

        # Classify the article
        classify_url = f"{API_BASE_URL}/api/classify/article/{article_id}?return_all_scores=true"

        response = requests.post(classify_url)

        if response.status_code == 200:
            data = response.json()
            print(f"\n[+] Classification successful")
            print(f"Topic: {data.get('topic')}")
            print(f"Confidence: {data.get('confidence'):.3f}")

            if 'all_scores' in data and data['all_scores']:
                print("\nAll Scores:")
                for topic, score in sorted(
                    data['all_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    print(f"  {topic}: {score:.3f}")

            return True
        else:
            print(f"[-] Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print_section("TEST 6: Error Handling")

    tests_passed = 0
    tests_total = 3

    # Test 1: Empty text
    print("\n1. Testing empty text...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/classify/text",
            json={"text": ""}
        )
        # Should either return 422 (validation error) or handle gracefully
        if response.status_code in [422, 200]:
            print("   [+] Handled correctly")
            tests_passed += 1
        else:
            print(f"   [-] Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 2: Invalid article ID
    print("\n2. Testing invalid article ID...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/classify/article/999999"
        )
        if response.status_code == 404:
            print("   [+] Correctly returned 404")
            tests_passed += 1
        else:
            print(f"   [-] Expected 404, got {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 3: Too many texts in batch
    print("\n3. Testing batch size limit...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/classify/batch",
            json={"texts": ["test"] * 101}  # Over the 100 limit
        )
        # Should return 422 validation error
        if response.status_code == 422:
            print("   [+] Correctly rejected oversized batch")
            tests_passed += 1
        else:
            print(f"   [-] Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    print(f"\nError handling: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print(" CLASSIFICATION API TEST SUITE")
    print("="*70)

    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("\n[-] API server is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("\n[-] Cannot connect to API server at", API_BASE_URL)
        print("\nPlease start the server first:")
        print("  uvicorn backend.api.main:app --reload")
        return False

    print(f"\n[+] API server is running at {API_BASE_URL}")

    results = []

    # Run tests
    results.append(("Model Info", test_model_info()))
    results.append(("Classify Text", test_classify_text()))
    results.append(("Classify with All Scores", test_classify_text_with_all_scores()))
    results.append(("Batch Classification", test_batch_classify()))

    # Optional test (skip if no articles)
    article_result = test_classify_article()
    if article_result is not None:
        results.append(("Classify Article by ID", article_result))

    results.append(("Error Handling", test_error_handling()))

    # Summary
    print_section("SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\n{passed}/{total} test suites passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print(f"\n[FAILED] {total - passed} test suite(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)