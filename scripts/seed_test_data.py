"""
Script to seed the database with test data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.database import SessionLocal
from backend.db.models import User, Article, Interaction
from datetime import datetime, timezone
import hashlib

def create_test_user():
    """Create a test user"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == "test@example.com").first()
        if existing_user:
            print(f"User already exists with ID: {existing_user.id}")
            return existing_user.id
        
        # Create new user
        user = User(
            email="test@example.com",
            hashed_password=hashlib.sha256("password123".encode()).hexdigest(),
            created_at=datetime.now(timezone.utc)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created user with ID: {user.id}")
        return user.id
    finally:
        db.close()

def create_test_article():
    """Create a test article if none exist"""
    db = SessionLocal()
    try:
        # Check if articles exist
        article_count = db.query(Article).count()
        if article_count > 0:
            print(f"Database already has {article_count} articles")
            # Get the first article
            first_article = db.query(Article).first()
            return first_article.id
        
        # Create a test article
        article = Article(
            url="https://example.com/test-article",
            title="Test Article",
            content="This is a test article for testing purposes.",
            source="example.com",
            published_at=datetime.now(timezone.utc),
            scraped_at=datetime.now(timezone.utc),
            topic="tech",
            topic_confidence=0.95
        )
        db.add(article)
        db.commit()
        db.refresh(article)
        print(f"Created test article with ID: {article.id}")
        return article.id
    finally:
        db.close()

def create_test_interaction(user_id, article_id):
    """Create a test interaction"""
    db = SessionLocal()
    try:
        interaction = Interaction(
            user_id=user_id,
            article_id=article_id,
            interaction_type="click",
            timestamp=datetime.now(timezone.utc),
            read_time_seconds=30
        )
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        print(f"Created interaction with ID: {interaction.id}")
        return interaction.id
    finally:
        db.close()

def main():
    """Main function to seed test data"""
    print("Seeding test data...")
    
    # Create test user
    user_id = create_test_user()
    
    # Create test article (if needed)
    article_id = create_test_article()
    
    # Create test interaction
    interaction_id = create_test_interaction(user_id, article_id)
    
    print("Test data seeding completed!")
    print(f"User ID: {user_id}")
    print(f"Article ID: {article_id}")
    print(f"Interaction ID: {interaction_id}")

if __name__ == "__main__":
    main()
