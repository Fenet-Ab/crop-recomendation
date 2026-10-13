import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
Base = declarative_base()

engine = None
SessionLocal = None


def get_engine():
    global engine, SessionLocal
    if engine is not None:
        return engine
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set in .env")
    connect_args = {}
    url = DATABASE_URL
    uses_supabase = "supabase" in url or "pooler.supabase.com" in url
    if uses_supabase:
        connect_args["sslmode"] = "require"
    # Transaction pooler (port 6543) should not use SQLAlchemy's default pooling.
    kwargs = {"pool_pre_ping": True, "connect_args": connect_args}
    if ":6543" in url:
        from sqlalchemy.pool import NullPool

        kwargs["poolclass"] = NullPool
    engine = create_engine(url, **kwargs)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine


def get_session():
    get_engine()
    if SessionLocal is None:
        raise RuntimeError("Database session factory was not created")
    return SessionLocal()


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()
