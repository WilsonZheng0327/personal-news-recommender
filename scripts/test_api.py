"""
Test script for API endpoints
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("SUCCESS: Health check passed")


def test_root():
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("SUCCESS: Root endpoint works")


def test_get_articles():
    """Test getting articles"""
    response = requests.get(f"{BASE_URL}/api/articles")
    assert response.status_code == 200
    articles = response.json()
    print(f"SUCCESS: Found {len(articles)} articles")
    return articles


def test_get_articles_with_filters():
    """Test article filtering"""
    # Test pagination
    response = requests.get(f"{BASE_URL}/api/articles?skip=0&limit=5")
    assert response.status_code == 200
    articles = response.json()
    assert len(articles) <= 5
    print(f"SUCCESS: Pagination works (got {len(articles)} articles)")


def test_article_count():
    """Test article count endpoint"""
    response = requests.get(f"{BASE_URL}/api/articles/count")
    assert response.status_code == 200
    data = response.json()
    print(f"SUCCESS: Total articles in DB: {data['count']}")


def test_recent_articles():
    """Test recent articles endpoint"""
    response = requests.get(f"{BASE_URL}/api/articles/recent?limit=5")
    assert response.status_code == 200
    articles = response.json()
    print(f"SUCCESS: Got {len(articles)} recent articles")


def test_get_single_article(article_id):
    """Test getting single article"""
    response = requests.get(f"{BASE_URL}/api/articles/{article_id}")
    assert response.status_code == 200
    article = response.json()
    print(f"SUCCESS: Retrieved article: {article['title'][:50]}...")
    assert "content" in article  # DetailResponse has content
    return article


def test_404_error():
    """Test 404 error handling"""
    response = requests.get(f"{BASE_URL}/api/articles/99999")
    assert response.status_code == 404
    error = response.json()
    assert "detail" in error
    print("SUCCESS: 404 error handling works")


def test_log_interaction():
    """Test logging interaction"""
    payload = {
        "user_id": 1,
        "article_id": 1,
        "interaction_type": "click",
        "read_time_seconds": 45
    }
    response = requests.post(f"{BASE_URL}/api/interaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    print("SUCCESS: Interaction logged successfully")


def test_get_user_interactions():
    """Test getting user interactions"""
    response = requests.get(f"{BASE_URL}/api/interactions/1")
    assert response.status_code == 200
    interactions = response.json()
    print(f"SUCCESS: User has {len(interactions)} interactions")


def test_stats():
    """Test stats endpoint"""
    response = requests.get(f"{BASE_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"SUCCESS: System stats:")
    print(f"  - Articles: {stats['total_articles']}")
    print(f"  - Users: {stats['total_users']}")
    print(f"  - Interactions: {stats['total_interactions']}")
    if stats['topic_distribution']:
        print(f"  - Topics: {stats['topic_distribution']}")


def run_all_tests():
    """Run all tests"""
    print("Starting API tests...\n")
    
    try:
        # Health checks
        test_health()
        test_root()
        print()
        
        # Article endpoints
        print("Testing article endpoints...")
        articles = test_get_articles()
        test_get_articles_with_filters()
        test_article_count()
        test_recent_articles()
        
        if articles:
            test_get_single_article(articles[0]['id'])
        
        test_404_error()
        print()
        
        # Interaction endpoints
        print("Testing interaction endpoints...")
        test_log_interaction()
        test_get_user_interactions()
        print()
        
        # Stats
        print("Testing stats endpoint...")
        test_stats()
        print()
        
        print("SUCCESS: All tests passed!")
        
    except AssertionError as e:
        print(f"\nERROR: Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API. Is it running?")
        print("   Run: uvicorn backend.api.main:app --reload")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()