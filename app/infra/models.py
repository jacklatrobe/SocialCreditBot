"""
Database models and schemas for Discord Observer/Orchestrator.
SQLAlchemy models with async SQLite support.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, DateTime, Integer, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.sql import func

Base = declarative_base()


class SignalRecord(Base):
    """Database model for Signal storage."""
    __tablename__ = "signals"

    # Primary key and basic fields
    signal_id = Column(String(255), primary_key=True)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Author and context as JSON
    author = Column(JSON, nullable=False)
    context = Column(JSON, nullable=False)
    
    # Message content
    content = Column(Text, nullable=False)
    
    # Analysis results as JSON fields
    nano_tags = Column(JSON, nullable=True)
    observer = Column(JSON, nullable=True)
    orchestrator = Column(JSON, nullable=True)
    signal_metadata = Column(JSON, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index('idx_signals_source_created', 'source', 'created_at'),
        Index('idx_signals_updated', 'updated_at'),
        Index('idx_signals_context', 'context'),  # For channel/guild queries
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'signal_id': self.signal_id,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'author': self.author,
            'context': self.context,
            'content': self.content,
            'nano_tags': self.nano_tags,
            'observer': self.observer,
            'orchestrator': self.orchestrator,
            'metadata': self.signal_metadata
        }


class ActionRecord(Base):
    """Database model for action logging."""
    __tablename__ = "actions"

    # Primary key
    action_id = Column(String(255), primary_key=True)
    signal_id = Column(String(255), nullable=False)
    
    # Action details
    action = Column(String(100), nullable=False)  # e.g., "discord.reply"
    request = Column(JSON, nullable=False)        # Request data
    response = Column(JSON, nullable=True)        # Response data
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Status tracking
    status = Column(String(50), nullable=False, default="pending")  # pending, success, failed, retry
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Indexes for performance
    __table_args__ = (
        Index('idx_actions_signal_id', 'signal_id'),
        Index('idx_actions_action_created', 'action', 'created_at'),
        Index('idx_actions_status', 'status'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'action_id': self.action_id,
            'signal_id': self.signal_id,
            'action': self.action,
            'request': self.request,
            'response': self.response,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


class KeyValueRecord(Base):
    """Simple key-value store for system state and configuration."""
    __tablename__ = "kv_store"

    key = Column(String(255), primary_key=True)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Optional metadata
    kv_metadata = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.kv_metadata
        }