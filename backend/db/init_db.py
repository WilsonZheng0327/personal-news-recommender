import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import engine
from backend.db.models import Base

def init_database():
    """Create all tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("SUCCESS: Database initialized successfully!")

if __name__ == "__main__":
    init_database()