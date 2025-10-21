import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create scripts/test_connection.py
from backend.db.database import engine
from sqlalchemy import text

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            print(f"Result: {result.scalar()}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")


import redis
from config.settings import get_settings

def test_redis():
    settings = get_settings()
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test set/get
        r.set("test_key", "Hello Redis!")
        value = r.get("test_key")
        print(f"Test value: {value.decode('utf-8')}")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")

if __name__ == "__main__":
    test_connection()
    test_redis()
