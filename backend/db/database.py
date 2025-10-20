from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import get_settings

settings = get_settings()

"""
Keeps some # of connections open all the time
    when needed, can have some overflow
    beyond overflow, will have to wait in queue
"""
# Create engine
engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,     # Verify connections before using
    pool_size=10,           # SQLAlchemy maintains a "pool" of open connections
    max_overflow=20         # Allows 20 more open connections on top of pool
)

"""
Session objects manage database operations
- it will grab a connection when operations are called
- returns the connection after use, doesn't close it
"""
# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,       # Must call db.commit() to save changes
    autoflush=False,        # Must call db.commit() to sync changes to database
    bind=engine             # Sessions created will use the db engine
)

# Base class for models
Base = declarative_base()

def get_db():
    """Dependency for FastAPI routes"""
    db = SessionLocal()     # Create new session
    try:    
        yield db            # Give it to the route
                            # yield so when the caller finishes, auto-closes
    finally:
        db.close()          # Close when done