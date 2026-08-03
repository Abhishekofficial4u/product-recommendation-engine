"""
backend/api/database.py
========================
SQLite database setup using SQLAlchemy.
Stores two tables:
  - ratings_log : every rating submitted via POST /rate
  - request_log : every recommendation request (for analytics)
"""

from sqlalchemy import create_engine, Integer, Float, String, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "rec_engine.db")
DB_URL   = f"sqlite:///{DB_PATH}"

engine        = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal  = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models in modern SQLAlchemy 2.0 style."""
    pass


class User(Base):
    """Stores user accounts for authentication."""
    __tablename__ = "users"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str]        = mapped_column(String(120), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RatingLog(Base):
    """Stores every new rating submitted by users via POST /rate."""
    __tablename__ = "ratings_log"

    id: Mapped[int]         = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int]    = mapped_column(Integer, nullable=False, index=True)
    item_id: Mapped[int]    = mapped_column(Integer, nullable=False, index=True)
    rating: Mapped[float]   = mapped_column(Float,   nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RequestLog(Base):
    """Logs every recommendation request for analytics."""
    __tablename__ = "request_log"

    id: Mapped[int]           = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int]      = mapped_column(Integer, nullable=False, index=True)
    model_used: Mapped[str]   = mapped_column(String(32), nullable=False)
    top_k: Mapped[int]        = mapped_column(Integer, nullable=False)
    response_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)



def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
