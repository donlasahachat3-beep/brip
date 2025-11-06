"""SQLAlchemy models for target management."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Target(Base):
    __tablename__ = "targets"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    url = Column(Text, nullable=False)
    tags = Column(Text, nullable=True)
    credentials = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
