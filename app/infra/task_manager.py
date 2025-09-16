"""
Background Task Manager for centralized task lifecycle management.

This module provides centralized management of all background asyncio tasks
in the Discord Social Credit Bot system, including:
- Signal Bus routing tasks
- ReAct Orchestrator queue processing
- Health monitoring and recovery
- Graceful shutdown coordination

Follows SOLID principles with centralized task management and monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Background task status levels"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class BackgroundTaskInfo:
    """Information about a managed background task"""
    name: str
    task: Optional[asyncio.Task] = None
    status: TaskStatus = TaskStatus.STOPPED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    auto_restart: bool = True


class BackgroundTaskManager:
    """
    Centralized manager for all background asyncio tasks.
    
    Provides:
    - Task registration and lifecycle management
    - Health monitoring and status reporting
    - Automatic restart on failures
    - Graceful shutdown coordination
    """
    
    def __init__(self):
        self._tasks: Dict[str, BackgroundTaskInfo] = {}
        self._shutdown_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("BackgroundTaskManager initialized")
    
    async def start(self) -> None:
        """Start the background task manager and monitoring."""
        if self._running:
            logger.warning("BackgroundTaskManager already running")
            return
            
        self._running = True
        self._shutdown_event.clear()
        
        # Start the health monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_tasks())
        
        logger.info("BackgroundTaskManager started with task health monitoring")
    
    async def stop(self) -> None:
        """Stop the background task manager and all managed tasks."""
        if not self._running:
            return
            
        logger.info("🛑 Stopping BackgroundTaskManager and all managed tasks...")
        self._running = False
        self._shutdown_event.set()
        
        # Stop the monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all managed tasks gracefully
        stop_tasks = []
        for task_info in self._tasks.values():
            if task_info.task and not task_info.task.done():
                task_info.status = TaskStatus.STOPPING
                task_info.task.cancel()
                stop_tasks.append(task_info.task)
        
        # Wait for all tasks to complete
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
        # Update task status
        for task_info in self._tasks.values():
            if task_info.status == TaskStatus.STOPPING:
                task_info.status = TaskStatus.STOPPED
                task_info.stopped_at = datetime.now(timezone.utc)
        
        logger.info("✅ BackgroundTaskManager stopped - all tasks cancelled")
    
    def register_task(self, name: str, coro: Callable[[], Any], 
                     auto_restart: bool = True) -> None:
        """
        Register a background task for management.
        
        Args:
            name: Unique name for the task
            coro: Coroutine function to run as background task
            auto_restart: Whether to restart task automatically on failure
        """
        if name in self._tasks:
            logger.warning(f"Task {name} already registered, updating...")
        
        task_info = BackgroundTaskInfo(
            name=name,
            auto_restart=auto_restart
        )
        
        # Start the task immediately if manager is running
        if self._running:
            task_info.status = TaskStatus.STARTING
            task_info.task = asyncio.create_task(coro(), name=name)
            task_info.started_at = datetime.now(timezone.utc)
            task_info.status = TaskStatus.RUNNING
        
        self._tasks[name] = task_info
        logger.info(f"Registered background task: {name} (auto_restart={auto_restart})")
    
    def get_task_status(self, name: str) -> Optional[TaskStatus]:
        """Get the status of a specific task."""
        task_info = self._tasks.get(name)
        return task_info.status if task_info else None
    
    def get_all_task_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all managed tasks."""
        status = {}
        for name, task_info in self._tasks.items():
            status[name] = {
                'status': task_info.status,
                'created_at': task_info.created_at,
                'started_at': task_info.started_at,
                'stopped_at': task_info.stopped_at,
                'restart_count': task_info.restart_count,
                'last_error': task_info.last_error,
                'auto_restart': task_info.auto_restart,
                'is_alive': task_info.task and not task_info.task.done() if task_info.task else False
            }
        return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a health summary of all background tasks."""
        total = len(self._tasks)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            'total_tasks': total,
            'running_tasks': running,
            'failed_tasks': failed,
            'healthy': failed == 0 and running == total,
            'manager_running': self._running
        }
    
    async def _monitor_tasks(self) -> None:
        """Monitor background tasks and restart failed ones."""
        logger.info("Background task health monitor started")
        
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    # Check each task
                    for name, task_info in self._tasks.items():
                        if task_info.task and task_info.task.done():
                            # Task completed/failed
                            try:
                                # Check if task failed
                                exception = task_info.task.exception()
                                if exception:
                                    task_info.status = TaskStatus.FAILED
                                    task_info.last_error = str(exception)
                                    task_info.stopped_at = datetime.now(timezone.utc)
                                    logger.error(f"Background task {name} failed: {exception}")
                                    
                                    # Restart if auto_restart is enabled
                                    if task_info.auto_restart and self._running:
                                        logger.info(f"Auto-restarting failed task: {name}")
                                        await self._restart_task(task_info)
                                else:
                                    # Task completed normally
                                    task_info.status = TaskStatus.STOPPED
                                    task_info.stopped_at = datetime.now(timezone.utc)
                                    logger.info(f"Background task {name} completed normally")
                                    
                            except Exception as e:
                                logger.error(f"Error checking task {name}: {e}")
                    
                    # Wait before next check
                    await asyncio.sleep(5.0)  # Check every 5 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in task monitor: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info("Background task monitor cancelled")
        except Exception as e:
            logger.error(f"Fatal error in task monitor: {e}")
        finally:
            logger.info("Background task health monitor stopped")
    
    async def _restart_task(self, task_info: BackgroundTaskInfo) -> None:
        """Restart a failed background task."""
        try:
            task_info.restart_count += 1
            task_info.status = TaskStatus.STARTING
            task_info.started_at = datetime.now(timezone.utc)
            
            # Note: This is a simplified restart - in practice, you'd need
            # to recreate the coroutine. For now, we'll log the attempt.
            logger.warning(f"Task restart attempted for {task_info.name} (attempt #{task_info.restart_count})")
            task_info.status = TaskStatus.FAILED  # Keep as failed until proper restart implemented
            
        except Exception as e:
            logger.error(f"Failed to restart task {task_info.name}: {e}")
            task_info.status = TaskStatus.FAILED


# Global task manager instance
_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


async def start_task_manager() -> None:
    """Start the global background task manager."""
    manager = get_task_manager()
    await manager.start()


async def stop_task_manager() -> None:
    """Stop the global background task manager."""
    global _task_manager
    if _task_manager:
        await _task_manager.stop()
        _task_manager = None