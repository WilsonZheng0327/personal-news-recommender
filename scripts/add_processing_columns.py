"""
Database Migration: Add processing columns to articles table

This script adds the new processing-related columns to existing articles table
WITHOUT losing any existing data.

Columns to add:
- processing_status (default: 'pending')
- processed_at (nullable timestamp)
- processing_error (nullable text)
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.database import SessionLocal
from sqlalchemy import text

print("="*70)
print("DATABASE MIGRATION: Add Processing Columns")
print("="*70)

db = SessionLocal()

try:
    print("\n1. Checking if columns already exist...")

    # Check if processing_status exists
    result = db.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'articles' AND column_name = 'processing_status';
    """))

    if result.fetchone():
        print("   [INFO] Columns already exist! Skipping migration.")
        db.close()
        sys.exit(0)

    print("   [INFO] Columns do not exist. Proceeding with migration...")

    # Add the three columns
    print("\n2. Adding 'processing_status' column...")
    db.execute(text("""
        ALTER TABLE articles
        ADD COLUMN processing_status VARCHAR DEFAULT 'pending';
    """))
    print("   [OK] Added processing_status column")

    print("\n3. Adding 'processed_at' column...")
    db.execute(text("""
        ALTER TABLE articles
        ADD COLUMN processed_at TIMESTAMP WITHOUT TIME ZONE;
    """))
    print("   [OK] Added processed_at column")

    print("\n4. Adding 'processing_error' column...")
    db.execute(text("""
        ALTER TABLE articles
        ADD COLUMN processing_error TEXT;
    """))
    print("   [OK] Added processing_error column")

    # Commit the changes
    print("\n5. Committing changes...")
    db.commit()
    print("   [OK] Migration complete!")

    # Verify
    print("\n6. Verifying migration...")
    result = db.execute(text("""
        SELECT column_name, data_type, column_default
        FROM information_schema.columns
        WHERE table_name = 'articles'
          AND column_name IN ('processing_status', 'processed_at', 'processing_error')
        ORDER BY ordinal_position;
    """))

    print("\n   Column Name              | Type             | Default")
    print("   " + "-"*60)

    for row in result:
        col_name = row[0]
        data_type = row[1]
        default = row[2] if row[2] else "None"
        print(f"   {col_name:24} | {data_type:16} | {default}")

    # Check article counts by status
    print("\n7. Checking article processing status...")
    result = db.execute(text("""
        SELECT processing_status, COUNT(*)
        FROM articles
        GROUP BY processing_status;
    """))

    print("\n   Status      | Count")
    print("   " + "-"*25)

    for row in result:
        status = row[0]
        count = row[1]
        print(f"   {status:11} | {count}")

    print("\n" + "="*70)
    print("MIGRATION SUCCESSFUL!")
    print("="*70)
    print("\nAll existing articles have been set to 'pending' status.")
    print("They are ready to be processed by Celery tasks.")

except Exception as e:
    print(f"\n[ERROR] Migration failed: {e}")
    print("Rolling back changes...")
    db.rollback()
    sys.exit(1)

finally:
    db.close()
