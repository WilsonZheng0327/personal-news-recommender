import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import feedparser
from newspaper import Article, Config
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import time
import random
from backend.db.database import SessionLocal
from backend.db.models import Article as ArticleModel

# RSS feeds to scrape
RSS_FEEDS = [
    # Easy to scrape (open content)
    "https://www.theguardian.com/world/rss",
    "https://www.npr.org/rss/rss.php?id=1001",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
    
    # Tech news (usually scraper-friendly)
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.engadget.com/rss.xml",
    
    # Reddit (very easy)
    "https://www.reddit.com/r/worldnews/.rss",
    "https://www.reddit.com/r/technology/.rss",
    "https://www.reddit.com/r/science/.rss",
    
    # BBC (sometimes works)
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    
    # Alternative news aggregators
    "https://news.ycombinator.com/rss",
]

# Configure newspaper to look like a real browser
def get_newspaper_config():
    """Create newspaper config with browser-like headers"""
    config = Config()
    
    # Mimic a real browser
    config.browser_user_agent = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
    
    # Request settings
    config.request_timeout = 10
    config.number_threads = 1
    config.fetch_images = False  # Faster scraping
    config.memoize_articles = False
    
    return config


def scrape_rss_feed(feed_url: str, db: Session):
    """Scrape a single RSS feed"""
    print(f"Scraping: {feed_url}")
    
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"  ❌ Error parsing feed: {e}")
        return
    
    if not feed.entries:
        print(f"  ⚠️  No entries found in feed")
        return
    
    articles_added = 0
    articles_skipped = 0
    articles_failed = 0
    
    config = get_newspaper_config()
    
    for idx, entry in enumerate(feed.entries):
        try:
            # Check if article already exists
            existing = db.query(ArticleModel).filter(ArticleModel.url == entry.link).first()
            if existing:
                articles_skipped += 1
                continue
            
            # Add random delay to avoid rate limiting (1-3 seconds)
            if idx > 0:
                time.sleep(random.uniform(1, 3))
            
            # Download full article content with config
            article = Article(entry.link, config=config)
            article.download()
            article.parse()
            
            # Validate that we got actual content
            if not article.text or len(article.text) < 100:
                print(f"  ⚠️  Insufficient content from: {entry.link[:60]}...")
                articles_failed += 1
                continue
            
            # Create database entry
            db_article = ArticleModel(
                url=entry.link,
                title=entry.title if hasattr(entry, 'title') else "No Title",
                content=article.text,
                source=feed_url,
                published_at=datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) if hasattr(entry, 'published_parsed') else None,
                author=article.authors[0] if article.authors else None,
                image_url=article.top_image if article.top_image else None
            )
            
            db.add(db_article)
            
            # Commit immediately to catch duplicates
            try:
                db.commit()
                articles_added += 1
                print(f"  ✓ Added: {entry.title[:60]}...")
            except IntegrityError:
                db.rollback()
                articles_skipped += 1
                
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ⚠️  Error scraping {entry.link[:60]}...: {error_msg}")
            articles_failed += 1
            db.rollback()
            continue
    
    # Print summary
    print(f"  📊 Summary - Added: {articles_added} | Skipped: {articles_skipped} | Failed: {articles_failed}\n")


def scrape_all_feeds():
    """Scrape all RSS feeds"""
    db = SessionLocal()
    try:
        print(f"\n🔄 Starting scrape of {len(RSS_FEEDS)} feeds...\n")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for feed_url in RSS_FEEDS:
            scrape_rss_feed(feed_url, db)
        
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"\n✅ Scraping complete in {elapsed:.1f} seconds!")
        
        # Show final count
        total = db.query(ArticleModel).count()
        print(f"📰 Total articles in database: {total}\n")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    scrape_all_feeds()