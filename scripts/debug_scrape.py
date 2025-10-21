import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from newspaper import Article, Config

def test_scrape(url):
    """Test scraping a single URL"""
    print(f"Testing: {url}\n")
    
    # Without config
    print("1. Testing without browser headers...")
    try:
        article = Article(url)
        article.download()
        article.parse()
        print(f"   Title: {article.title}")
        print(f"   Content length: {len(article.text)} chars")
        print(f"   Authors: {article.authors}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # With config
    print("2. Testing with browser headers...")
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        print(f"   Title: {article.title}")
        print(f"   Content length: {len(article.text)} chars")
        print(f"   First 200 chars: {article.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    # Test with a recent CNN article
    test_url = "https://www.cnn.com/2025/01/20/business/boeing-737-max-production/index.html"
    test_scrape(test_url)