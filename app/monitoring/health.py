"""
Health Check API Module

This module provides comprehensive health check endpoints for monitoring
the Discord Social Credit Bot system components including:

- System health (uptime, memory, resources)  
- Database connectivity (SQLite operations)
- Discord API availability 
- OpenAI LLM service health
- Signal bus and orchestrator status
- Component-level health checks

FastAPI-based endpoints following standard health check patterns
for container orchestration and monitoring systems.
"""

import asyncio
import logging
import os
import platform
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

import platform
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import Settings as Config
from app.infra.db import Database
from app.infra.bus import SignalBus, SignalType, BusMessage
from app.infra.task_manager import get_task_manager
from app.llm.client import OpenAILLMClient, LLMConfig
from app.orchestrator.core import MessageOrchestrator


class HealthStatus(str, Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Individual component health information"""
    name: str
    status: HealthStatus
    message: str
    last_checked: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SystemHealth(BaseModel):
    """System-wide health response model"""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str = "1.0.0"
    components: Dict[str, ComponentHealth]
    system_info: Dict[str, Any]


class HealthChecker:
    """
    Comprehensive system health checker following SOLID principles:
    
    - Single Responsibility: Each check method handles one component
    - Open/Closed: Easy to add new health checks by extending
    - Dependency Inversion: Depends on abstractions, not concrete classes
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self._db_manager: Optional[Database] = None
        self._signal_bus: Optional[SignalBus] = None
        self._llm_client: Optional[OpenAILLMClient] = None
        self._orchestrator: Optional[MessageOrchestrator] = None
    
    async def get_system_health(self) -> SystemHealth:
        """
        Get comprehensive system health status
        
        Follows Clean Code principles:
        - Small method with clear purpose
        - Descriptive naming
        - Minimal side effects
        """
        logger = logging.getLogger(__name__)
        logger.info("🔍 Starting comprehensive health check...")
        
        components = {}
        
        # Run all health checks in parallel for better performance
        health_checks = [
            self._check_system_resources(),
            self._check_database_health(),
            self._check_signal_bus_health(),
            self._check_background_tasks_health(),
            self._check_llm_service_health(),
            self._check_orchestrator_health(),
            self._check_training_data_health()
        ]
        
        try:
            logger.info("🔍 Running health checks for 7 components...")
            results = await asyncio.gather(*health_checks, return_exceptions=True)
            
            component_names = ["system", "database", "signal_bus", "background_tasks", "llm", "orchestrator", "training_data"]
            
            for i, result in enumerate(results):
                component_name = component_names[i]
                
                if isinstance(result, ComponentHealth):
                    components[result.name] = result
                    logger.info(f"✅ {component_name} health check: {result.status} - {result.message}")
                elif isinstance(result, Exception):
                    # Handle individual component failures gracefully
                    error_msg = f"Health check failed: {str(result)}"
                    logger.error(f"❌ {component_name} health check failed: {str(result)}")
                    components[component_name] = ComponentHealth(
                        name=component_name,
                        status=HealthStatus.UNHEALTHY,
                        message=error_msg,
                        last_checked=datetime.now()
                    )
        
        except Exception as e:
            # Fallback if gather itself fails
            logger.error(f"❌ Health check system error: {str(e)}")
            components["system"] = ComponentHealth(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check system error: {str(e)}",
                last_checked=datetime.now()
            )
        
        # Determine overall system status
        overall_status = self._determine_overall_status(components)
        logger.info(f"🏥 Overall system health: {overall_status}")
        
        # Log component summary
        healthy_count = sum(1 for c in components.values() if c.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED)  
        unhealthy_count = sum(1 for c in components.values() if c.status == HealthStatus.UNHEALTHY)
        
        logger.info(f"📊 Health summary: {healthy_count} healthy, {degraded_count} degraded, {unhealthy_count} unhealthy")
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self.start_time,
            components=components,
            system_info=self._get_system_info()
        )
    
    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource utilization"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Define thresholds for resource health
            cpu_warning_threshold = 80.0
            memory_warning_threshold = 85.0
            disk_warning_threshold = 90.0
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > cpu_warning_threshold:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > memory_warning_threshold:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                
            if disk.percent > disk_warning_threshold:
                status = HealthStatus.DEGRADED  
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "System resources within normal limits"
            if issues:
                message = f"Resource warnings: {', '.join(issues)}"
            
            return ComponentHealth(
                name="system",
                status=status,
                message=message,
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_database_health(self) -> ComponentHealth:
        """Check database connectivity and basic operations"""
        start_time = time.time()
        
        try:
            # Ensure the data directory exists and use absolute path
            db_path = self.config.db_path
            if not os.path.isabs(db_path):
                # Make relative path absolute from project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                db_path = os.path.join(project_root, db_path)
                
            if not self._db_manager:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                # Database constructor expects just the path, not the full URL
                self._db_manager = Database(db_path)
            
            # Test basic database operations
            await self._db_manager.initialize()
            
            # Simple query to test connectivity
            async with self._db_manager.get_session() as session:
                # Test query - just check if we can execute SQL
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                
                if row and row[0] == 1:
                    status = HealthStatus.HEALTHY
                    message = "Database connection and basic operations working"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "Database query returned unexpected result"
            
            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "database_path": db_path,
                    "driver": "aiosqlite"
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_signal_bus_health(self) -> ComponentHealth:
        """Check signal bus functionality"""
        start_time = time.time()
        
        try:
            if not self._signal_bus:
                self._signal_bus = SignalBus()
            
            # Test signal bus by creating and publishing a test message
            test_message = BusMessage(
                signal_type=SignalType.SIGNAL_TEST,
                data={"test": True, "health_check": True},
                timestamp=datetime.now(),
                source="health_checker",
                message_id="health_check_test",
                correlation_id=None
            )
            
            # Start signal bus if not running
            if not self._signal_bus.running:
                await self._signal_bus.start()
            
            # Test publishing (this should not raise exceptions)
            await self._signal_bus.publish(
                signal_type=SignalType.SIGNAL_TEST,
                data={"test": True, "health_check": True},
                source="health_checker"
            )
            
            return ComponentHealth(
                name="signal_bus",
                status=HealthStatus.HEALTHY,
                message="Signal bus operational and accepting messages",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "running": self._signal_bus.running,
                    "queue_size": self._signal_bus.message_queue.qsize() if self._signal_bus.message_queue else 0
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="signal_bus",
                status=HealthStatus.UNHEALTHY,
                message=f"Signal bus health check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_background_tasks_health(self) -> ComponentHealth:
        """Check background task manager and individual task status"""
        start_time = time.time()
        
        try:
            task_manager = get_task_manager()
            health_summary = task_manager.get_health_summary()
            all_task_status = task_manager.get_all_task_status()
            
            # Determine overall status
            if not health_summary['manager_running']:
                status = HealthStatus.UNHEALTHY
                message = "Background task manager is not running"
            elif health_summary['failed_tasks'] > 0:
                status = HealthStatus.DEGRADED
                message = f"{health_summary['failed_tasks']} of {health_summary['total_tasks']} background tasks failed"
            elif health_summary['healthy']:
                status = HealthStatus.HEALTHY
                message = f"All {health_summary['total_tasks']} background tasks running normally"
            else:
                status = HealthStatus.DEGRADED
                message = f"Background tasks partially operational ({health_summary['running_tasks']}/{health_summary['total_tasks']} running)"
            
            return ComponentHealth(
                name="background_tasks",
                status=status,
                message=message,
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    'summary': health_summary,
                    'task_details': all_task_status
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="background_tasks",
                status=HealthStatus.UNHEALTHY,
                message=f"Background task health check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )

    async def _check_llm_service_health(self) -> ComponentHealth:
        """Check LLM service configuration without making API calls"""
        start_time = time.time()
        
        try:
            # Check if LLM configuration is available
            if not hasattr(self.config, 'llm_api_key') or not self.config.llm_api_key:
                return ComponentHealth(
                    name="llm",
                    status=HealthStatus.DEGRADED,
                    message="LLM API key not configured",
                    last_checked=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={"config_missing": "llm_api_key"}
                )
            
            # Check if LLM model is configured
            if not hasattr(self.config, 'llm_model') or not self.config.llm_model:
                return ComponentHealth(
                    name="llm",
                    status=HealthStatus.DEGRADED,
                    message="LLM model not configured",
                    last_checked=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={"config_missing": "llm_model"}
                )
            
            # Check if client can be instantiated (no API call)
            if not self._llm_client:
                try:
                    self._llm_client = OpenAILLMClient()
                except Exception as e:
                    return ComponentHealth(
                        name="llm",
                        status=HealthStatus.DEGRADED,
                        message=f"LLM client initialization failed: {str(e)}",
                        last_checked=datetime.now(),
                        response_time_ms=(time.time() - start_time) * 1000,
                        details={"initialization_error": str(e)}
                    )
            
            # If we get here, configuration looks good
            return ComponentHealth(
                name="llm",
                status=HealthStatus.HEALTHY,
                message="LLM service configured and client ready (no API test performed)",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "model": getattr(self.config, 'llm_model', 'unknown'),
                    "api_key_configured": bool(getattr(self.config, 'llm_api_key', None)),
                    "client_initialized": self._llm_client is not None
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="llm",
                status=HealthStatus.DEGRADED,
                message=f"LLM service configuration check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_orchestrator_health(self) -> ComponentHealth:
        """Check orchestrator component health"""
        start_time = time.time()
        
        try:
            if not self._orchestrator:
                # MessageOrchestrator only takes max_concurrent_tasks parameter
                self._orchestrator = MessageOrchestrator(max_concurrent_tasks=5)
            
            # Check if orchestrator is properly initialized with aggregator
            # The orchestrator now uses an aggregation-based rules engine instead of static rules
            if hasattr(self._orchestrator, '_aggregator') and self._orchestrator._aggregator:
                status = HealthStatus.HEALTHY
                message = "Orchestrator operational with user profile aggregation rules engine"
                
                # Get orchestrator statistics if available
                stats = {}
                if hasattr(self._orchestrator, 'get_stats'):
                    stats = self._orchestrator.get_stats()
                
                # Get aggregator statistics 
                aggregator_stats = {}
                if hasattr(self._orchestrator._aggregator, 'get_stats'):
                    aggregator_stats = self._orchestrator._aggregator.get_stats()
                
                details = {
                    "aggregation_rules_loaded": True,
                    "aggregator_active": True,
                    "orchestrator_stats": stats,
                    "aggregator_stats": aggregator_stats
                }
            else:
                status = HealthStatus.UNHEALTHY
                message = "Orchestrator initialized but aggregator not available - app cannot function without rules engine"
                details = {"aggregation_rules_loaded": False}
            
            return ComponentHealth(
                name="orchestrator",
                status=status,
                message=message,
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details=details
            )
            
        except Exception as e:
            return ComponentHealth(
                name="orchestrator", 
                status=HealthStatus.UNHEALTHY,
                message=f"Orchestrator health check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _check_training_data_health(self) -> ComponentHealth:
        """Check training data collection health"""
        start_time = time.time()
        
        try:
            # Check if training data is enabled
            if not self.config.training_data_enabled:
                return ComponentHealth(
                    name="training_data",
                    status=HealthStatus.DEGRADED,
                    message="Training data collection is disabled",
                    last_checked=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={"enabled": False}
                )
            
            # Check training data file
            training_path = self.config.training_data_path
            if not os.path.isabs(training_path):
                # Make relative path absolute from project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                training_path = os.path.join(project_root, training_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(training_path), exist_ok=True)
            
            details = {
                "enabled": True,
                "file_path": training_path,
                "file_exists": os.path.exists(training_path)
            }
            
            if os.path.exists(training_path):
                try:
                    file_stats = os.stat(training_path)
                    details.update({
                        "file_size_bytes": file_stats.st_size,
                        "file_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "file_readable": os.access(training_path, os.R_OK),
                        "file_writable": os.access(training_path, os.W_OK)
                    })
                    
                    # Count lines in JSONL file for basic health check
                    try:
                        with open(training_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                        details["record_count"] = line_count
                    except Exception as e:
                        details["record_count_error"] = str(e)
                    
                    status = HealthStatus.HEALTHY
                    message = f"Training data collection operational - {details.get('record_count', 'unknown')} records"
                    
                except Exception as e:
                    status = HealthStatus.DEGRADED
                    message = f"Training data file exists but cannot read stats: {str(e)}"
            else:
                status = HealthStatus.HEALTHY  # File will be created on first write
                message = "Training data collection enabled - file will be created on first write"
            
            return ComponentHealth(
                name="training_data",
                status=status,
                message=message,
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details=details
            )
            
        except Exception as e:
            return ComponentHealth(
                name="training_data",
                status=HealthStatus.UNHEALTHY,
                message=f"Training data health check failed: {str(e)}",
                last_checked=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def _determine_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """
        Determine overall system health from component statuses
        
        YAGNI principle: Simple logic for now, can be enhanced later
        """
        if not components:
            return HealthStatus.UNHEALTHY
            
        statuses = [comp.status for comp in components.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "python_version": f"{platform.python_version()}",
            "platform": platform.system(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3)
        }


# FastAPI application for health checks
app = FastAPI(
    title="Discord Social Credit Bot - Health Check API",
    description="Health monitoring endpoints for the Discord Social Credit Bot system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global health checker instance
health_checker: Optional[HealthChecker] = None


@app.on_event("startup")
async def startup_event():
    """Initialize health checker on startup"""
    global health_checker
    config = Config()
    health_checker = HealthChecker(config)


@app.get("/health", response_model=SystemHealth)
async def health_check():
    """
    Basic health check endpoint
    
    Returns comprehensive system health status including:
    - Overall system status (healthy/degraded/unhealthy)
    - Individual component health
    - System resource utilization
    - Response times for each check
    """
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_system_health()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/health/simple")
async def simple_health():
    """
    Simple health check for load balancers
    
    Returns 200 OK if system is healthy or degraded, 503 only if unhealthy
    """
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_system_health()
        
        if health_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            return {
                "status": health_status.status.value.lower(), 
                "timestamp": health_status.timestamp
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail=f"System status: {health_status.status}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/health/detailed")
async def detailed_health():
    """
    Detailed health information including component details
    
    Provides comprehensive system information for debugging and monitoring
    """
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_system_health()
        
        # Convert ComponentHealth dataclasses to dicts for JSON serialization
        components_dict = {}
        for name, component in health_status.components.items():
            components_dict[name] = {
                "name": component.name,
                "status": component.status,
                "message": component.message,
                "last_checked": component.last_checked,
                "response_time_ms": component.response_time_ms,
                "details": component.details
            }
        
        return {
            "status": health_status.status,
            "timestamp": health_status.timestamp,
            "uptime_seconds": health_status.uptime_seconds,
            "version": health_status.version,
            "components": components_dict,
            "system_info": health_status.system_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")


@app.get("/status")
async def system_status():
    """Alias for /health/detailed for compatibility"""
    return await detailed_health()