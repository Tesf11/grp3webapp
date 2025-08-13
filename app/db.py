# app/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager

DB_URL = os.getenv("DB_URL", "sqlite:///app/data.db")

engine = create_engine(
    DB_URL,
    future=True,
    echo=False,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

@contextmanager
def get_session():
    """Usage: with get_session() as s: ..."""
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()

def create_all():
    # Import models so tables are registered on Base.metadata
    from app import models  # noqa
    Base.metadata.create_all(bind=engine)
