"""
Check database schema and data

This script connects to your PostgreSQL database and shows:
1. Current columns in the articles table
2. Sample data
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.database import SessionLocal, engine
from sqlalchemy import text

print("="*70)
print("DATABASE INSPECTOR")
print("="*70)

# Test 1: Check if we can connect
print("\n1. Testing database connection...")
try:
    db = SessionLocal()
    print("   [OK] Connected to database")
except Exception as e:
    print(f"   [FAIL] Could not connect: {e}")
    sys.exit(1)

# Test 2: Check articles table columns
print("\n2. Checking 'articles' table columns...")
try:
    result = db.execute(text("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'articles'
        ORDER BY ordinal_position;
    """))

    print("\n   Column Name              | Type             | Nullable | Default")
    print("   " + "-"*68)

    for row in result:
        col_name = row[0]
        data_type = row[1]
        nullable = row[2]
        default = row[3] if row[3] else "None"
        print(f"   {col_name:24} | {data_type:16} | {nullable:8} | {default[:20]}")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 3: Count articles
print("\n3. Checking article counts...")
try:
    result = db.execute(text("SELECT COUNT(*) FROM articles;"))
    count = result.scalar()
    print(f"   Total articles: {count}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 4: Sample data
print("\n4. Sample article data (first 3)...")
try:
    result = db.execute(text("""
        SELECT id, title, topic, processing_status
        FROM articles
        LIMIT 3;
    """))

    print("\n   ID | Title (first 50 chars)                           | Topic     | Status")
    print("   " + "-"*85)

    for row in result:
        article_id = row[0]
        title = row[1][:50] if row[1] else "N/A"
        topic = row[2] if row[2] else "N/A"
        status = row[3] if row[3] else "N/A"
        print(f"   {article_id:2} | {title:50} | {topic:9} | {status}")

except Exception as e:
    print(f"   [FAIL] Error querying data: {e}")
    print("   This is expected if processing_status column doesn't exist yet")


# Test 5: Get target article
print("\n5. Article by ID")
try:
    result = db.execute(text("SELECT * FROM articles WHERE articles.topic = 'Sci/Tech';"))
    row = result.fetchone()
    print(row)
except Exception as e:
    print(f"   [FAIL] Error querying data: {e}")
    print("   This is expected if processing_status column doesn't exist yet")


db.close()