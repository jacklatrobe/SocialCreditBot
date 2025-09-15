"""
Database connection and operations for Discord Observer/Orchestrator.
Async SQLite operations using SQLAlchemy and aiosqlite.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .models import Base, SignalRecord, ActionRecord, KeyValueRecord
from ..signals.base import Signal


logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Database operation error."""
    pass


class Database:
    """Async database operations for Discord Observer/Orchestrator."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = None
        self.SessionLocal = None
        
    async def initialize(self):
        """Initialize the database connection and create tables."""
        try:
            # Create async engine with SQLite
            # Using thread pool for SQLite async operations
            self.engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.db_path}",
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20  # 20 second timeout
                },
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        if not self.SessionLocal:
            raise DatabaseError("Database not initialized")
            
        async with self.SessionLocal() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    # Signal operations
    async def save_signal(self, signal: Signal) -> bool:
        """Save or update a signal in the database."""
        try:
            async with self.get_session() as session:
                # Check if signal already exists
                result = await session.execute(
                    select(SignalRecord).where(SignalRecord.signal_id == signal.signal_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing signal
                    await session.execute(
                        update(SignalRecord)
                        .where(SignalRecord.signal_id == signal.signal_id)
                        .values(
                            updated_at=datetime.utcnow(),
                            content=signal.content,
                            nano_tags=signal.nano_tags,
                            observer=signal.observer,
                            orchestrator=signal.orchestrator,
                            signal_metadata=signal.metadata
                        )
                    )
                else:
                    # Create new signal record
                    record = SignalRecord(
                        signal_id=signal.signal_id,
                        source=signal.source,
                        created_at=signal.created_at,
                        author=signal.author,
                        context=signal.context,
                        content=signal.content,
                        nano_tags=signal.nano_tags,
                        observer=signal.observer,
                        orchestrator=signal.orchestrator,
                        signal_metadata=signal.metadata
                    )
                    session.add(record)
                
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to save signal {signal.signal_id}: {e}")
            raise DatabaseError(f"Signal save failed: {e}")
    
    async def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a signal by ID."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    select(SignalRecord).where(SignalRecord.signal_id == signal_id)
                )
                record = result.scalar_one_or_none()
                return record.to_dict() if record else None
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get signal {signal_id}: {e}")
            raise DatabaseError(f"Signal retrieval failed: {e}")
    
    async def get_signals_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent signals by source."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    select(SignalRecord)
                    .where(SignalRecord.source == source)
                    .order_by(SignalRecord.created_at.desc())
                    .limit(limit)
                )
                records = result.scalars().all()
                return [record.to_dict() for record in records]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get signals by source {source}: {e}")
            raise DatabaseError(f"Signal query failed: {e}")
    
    # Action operations
    async def save_action(self, action_id: str, signal_id: str, action: str, 
                         request: Dict[str, Any], response: Optional[Dict[str, Any]] = None,
                         status: str = "pending") -> bool:
        """Save an action log entry."""
        try:
            async with self.get_session() as session:
                record = ActionRecord(
                    action_id=action_id,
                    signal_id=signal_id,
                    action=action,
                    request=request,
                    response=response,
                    status=status
                )
                session.add(record)
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to save action {action_id}: {e}")
            raise DatabaseError(f"Action save failed: {e}")
    
    async def update_action_status(self, action_id: str, status: str, 
                                 response: Optional[Dict[str, Any]] = None,
                                 error_message: Optional[str] = None) -> bool:
        """Update action status and response."""
        try:
            async with self.get_session() as session:
                update_data = {"status": status}
                if response:
                    update_data["response"] = response
                if error_message:
                    update_data["error_message"] = error_message
                    
                await session.execute(
                    update(ActionRecord)
                    .where(ActionRecord.action_id == action_id)
                    .values(**update_data)
                )
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to update action {action_id}: {e}")
            raise DatabaseError(f"Action update failed: {e}")
    
    async def get_actions_for_signal(self, signal_id: str) -> List[Dict[str, Any]]:
        """Get all actions for a signal."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    select(ActionRecord)
                    .where(ActionRecord.signal_id == signal_id)
                    .order_by(ActionRecord.created_at.asc())
                )
                records = result.scalars().all()
                return [record.to_dict() for record in records]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get actions for signal {signal_id}: {e}")
            raise DatabaseError(f"Action query failed: {e}")
    
    # Key-value operations
    async def set_kv(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a key-value pair."""
        try:
            async with self.get_session() as session:
                # Check if key exists
                result = await session.execute(
                    select(KeyValueRecord).where(KeyValueRecord.key == key)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing
                    await session.execute(
                        update(KeyValueRecord)
                        .where(KeyValueRecord.key == key)
                        .values(
                            value=value,
                            updated_at=datetime.utcnow(),
                            kv_metadata=metadata
                        )
                    )
                else:
                    # Create new
                    record = KeyValueRecord(
                        key=key,
                        value=value,
                        kv_metadata=metadata
                    )
                    session.add(record)
                
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise DatabaseError(f"KV set failed: {e}")
    
    async def get_kv(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    select(KeyValueRecord).where(KeyValueRecord.key == key)
                )
                record = result.scalar_one_or_none()
                return record.value if record else None
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get key {key}: {e}")
            raise DatabaseError(f"KV get failed: {e}")
    
    async def delete_kv(self, key: str) -> bool:
        """Delete a key-value pair."""
        try:
            async with self.get_session() as session:
                await session.execute(
                    delete(KeyValueRecord).where(KeyValueRecord.key == key)
                )
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete key {key}: {e}")
            raise DatabaseError(f"KV delete failed: {e}")
    
    # Maintenance operations
    async def cleanup_old_signals(self, days_old: int = 14) -> int:
        """Clean up signals older than specified days."""
        try:
            async with self.get_session() as session:
                cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_old)
                
                # First get count
                count_result = await session.execute(
                    select(SignalRecord).where(SignalRecord.created_at < cutoff_date)
                )
                count = len(count_result.scalars().all())
                
                # Delete old signals
                await session.execute(
                    delete(SignalRecord).where(SignalRecord.created_at < cutoff_date)
                )
                
                # Delete associated actions
                await session.execute(
                    delete(ActionRecord).where(ActionRecord.created_at < cutoff_date)
                )
                
                await session.commit()
                logger.info(f"Cleaned up {count} old signals and their actions")
                return count
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old signals: {e}")
            raise DatabaseError(f"Cleanup failed: {e}")


# Global database instance
db = None


async def init_database(db_path: str) -> Database:
    """Initialize the global database instance."""
    global db
    db = Database(db_path)
    await db.initialize()
    return db


async def get_database() -> Database:
    """Get the global database instance."""
    if db is None:
        raise DatabaseError("Database not initialized")
    return db


async def close_database():
    """Close the global database instance."""
    global db
    if db:
        await db.close()
        db = None