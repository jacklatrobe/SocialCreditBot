#!/usr/bin/env python3
"""
entry point for the Discord Social Credit Bot system. It manages all
background services following Clean Architecture principles:

- Single Responsibility: FastAPI handles HTTP API, background services handle business logic
- Dependency Inversion: All services depend on shared infrastructure abstractions
- Open/Closed: Easy to add new API endpoints or background services

The Discord bot runs as a background service managed by FastAPI's lifespan events.
This provides proper container lifecycle management and simple HTTP health checks.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime

from app.config import get_settings, validate_configuration
from app.infra.db import init_database, close_database
from app.infra.bus import init_signal_bus, close_signal_bus, SignalType
from app.infra.task_manager import start_task_manager, stop_task_manager, get_task_manager
from app.llm.client import get_llm_client
from app.orchestrator.react_orchestrator import start_react_orchestrator, stop_react_orchestrator
from app.observer.llm_observer import start_llm_observer, stop_llm_observer
from app.ingest.discord_client import get_discord_bot
from app.monitoring.health import HealthChecker, SystemHealth, HealthStatus
from app.orchestrator.aggregator import get_aggregator
from app.infra.bus import get_signal_bus
from app.signals.discord import DiscordMessage

# Configure logging
log_handlers = [logging.StreamHandler(sys.stdout)]

# Try to add file handler, but don't fail if we can't write to the file
try:
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    log_handlers.append(logging.FileHandler('logs/discord_bot.log', encoding='utf-8'))
except (PermissionError, OSError) as e:
    # In containers, we might not have write permissions, so just log to stdout
    print(f"Warning: Could not create log file: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Global application state
app_state: Dict[str, Any] = {
    "config": None,
    "health_checker": None,
    "discord_bot": None,
    "background_tasks": set()
}


async def setup_system():
    """Setup all system components."""
    logger.info("🔧 Setting up system components...")
    
    try:
        # Load and validate configuration
        config = get_settings()
        validate_configuration()
        app_state["config"] = config
        
        logger.info(f"� Configuration: LLM={config.llm_model}, Training Data={'ON' if config.training_data_enabled else 'OFF'}")
        
        # Initialize database
        await init_database(config.db_path)
        logger.info("✅ Database initialized")
        
        # Initialize signal bus
        await init_signal_bus()
        logger.info("✅ Signal bus initialized")
        
        # Start centralized background task manager
        await start_task_manager()
        logger.info("✅ Background task manager started")
        
        # Initialize LLM client  
        await get_llm_client()
        logger.info("✅ LLM client initialized")
        
        # Start orchestrator
        await start_react_orchestrator()
        logger.info("✅ Orchestrator started")
        
        # Start LLM observer
        await start_llm_observer()
        logger.info("✅ LLM observer started")
        
        # Initialize health checker
        app_state["health_checker"] = HealthChecker(config)
        logger.info("✅ Health checker initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup system: {e}")
        return False


async def start_discord_bot():
    """Start Discord bot as a background task."""
    config = app_state["config"]
    if not config.discord_bot_token:
        logger.error("❌ Discord bot token not configured")
        return
    
    try:
        logger.info("🤖 Starting Discord bot...")
        bot = get_discord_bot()
        app_state["discord_bot"] = bot
        
        # Start the bot - this will run indefinitely
        await bot.start(config.discord_bot_token)
        
    except Exception as e:
        logger.error(f"❌ Discord bot error: {e}")
        raise


async def shutdown_system():
    """Shutdown all system components gracefully."""
    logger.info("🛑 Shutting down system...")
    
    try:
        # Stop Discord bot
        if app_state.get("discord_bot"):
            logger.info("🛑 Stopping Discord bot...")
            await app_state["discord_bot"].close()
            
        # Cancel background tasks
        for task in app_state["background_tasks"]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop system components
        await stop_task_manager()
        await stop_react_orchestrator()
        await stop_llm_observer()  
        await close_signal_bus()
        await close_database()
        
        logger.info("✅ System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown."""
    
    # Startup
    logger.info("🚀 Starting Discord Social Credit Bot API Server...")
    
    if not await setup_system():
        logger.error("❌ Failed to setup system components")
        sys.exit(1)
    
    # Start Discord bot as background task
    discord_task = asyncio.create_task(start_discord_bot())
    app_state["background_tasks"].add(discord_task)
    
    logger.info("✅ All systems operational")
    
    try:
        yield
    finally:
        # Shutdown
        await shutdown_system()


# Create FastAPI application
app = FastAPI(
    title="Discord Social Credit Bot API",
    description="Management and monitoring API for the Discord Social Credit Bot system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Pydantic models for API
class SimulateMessageRequest(BaseModel):
    content: str
    user_id: str
    username: str
    guild_id: str = "123456789"
    channel_id: str = "987654321"


# Health Check Endpoints
@app.get("/health", response_model=SystemHealth)
async def health_check():
    """
    Comprehensive health check endpoint for container orchestration.
    
    Returns detailed system health status including all components.
    """
    health_checker = app_state.get("health_checker")
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_system_health()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/health/simple")
async def simple_health():
    """
    Simple health check for load balancers and container health checks.
    
    Returns 200 OK if system is healthy, 503 if degraded/unhealthy.
    """
    health_checker = app_state.get("health_checker")
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_system_health()
        
        if health_status.status == HealthStatus.HEALTHY:
            return {"status": "healthy", "timestamp": health_status.timestamp}
        else:
            raise HTTPException(
                status_code=503, 
                detail=f"System status: {health_status.status}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# User Profile Endpoints  
@app.get("/profiles/stats")
async def get_profile_stats():
    """Get aggregator statistics and overview."""
    try:
        aggregator = get_aggregator()
        stats = aggregator.get_stats()
        return {
            "aggregator_stats": stats,
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get profile stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile stats: {str(e)}")


@app.get("/profiles/{user_id}")
async def get_user_profile(user_id: str):
    """Get detailed profile for a specific user."""
    try:
        aggregator = get_aggregator()
        profile = await aggregator.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"User profile not found: {user_id}")
        
        return {
            "user_id": user_id,
            "profile": {
                "total_messages": profile.total_messages,
                "first_seen": profile.first_seen.isoformat() if profile.first_seen else None,
                "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
                "is_new_user": profile.is_new_user,
                "is_frequent_questioner": profile.is_frequent_questioner,
                "is_problematic_user": profile.is_problematic_user,
                "requires_monitoring": profile.requires_monitoring,
                "unanswered_questions": profile.unanswered_questions,
                "unresolved_complaints": profile.unresolved_complaints,
                "recent_urgency_score": profile.recent_urgency_score,
                "recent_toxicity_score": profile.recent_toxicity_score
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")


@app.get("/profiles")
async def list_user_profiles(limit: int = 50):
    """List all user profiles with basic stats."""
    try:
        aggregator = get_aggregator()
        
        # Get all profiles (limited for performance)
        all_profiles = []
        profile_count = 0
        
        for user_id, profile in aggregator._profiles.items():
            if profile_count >= limit:
                break
                
            all_profiles.append({
                "user_id": user_id,
                "total_messages": profile.total_messages,
                "first_seen": profile.first_seen.isoformat() if profile.first_seen else None,
                "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
                "is_new_user": profile.is_new_user,
                "is_frequent_questioner": profile.is_frequent_questioner,
                "is_problematic_user": profile.is_problematic_user,
                "requires_monitoring": profile.requires_monitoring,
                "unanswered_questions": profile.unanswered_questions,
                "unresolved_complaints": profile.unresolved_complaints
            })
            profile_count += 1
        
        return {
            "profiles": all_profiles,
            "total_shown": len(all_profiles),
            "limit_applied": limit,
            "stats": aggregator.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to list user profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list user profiles: {str(e)}")


@app.get("/status")
async def system_status():
    """System status endpoint providing operational information."""
    return {
        "status": "operational",
        "services": {
            "discord_bot": "running" if app_state.get("discord_bot") else "stopped",
            "health_checker": "running" if app_state.get("health_checker") else "stopped",
            "background_tasks": len(app_state.get("background_tasks", set()))
        },
        "config": {
            "llm_model": app_state["config"].llm_model if app_state.get("config") else None,
            "training_data_enabled": app_state["config"].training_data_enabled if app_state.get("config") else None
        }
    }


@app.get("/")
async def root():
    """Root endpoint with basic system information."""
    return {
        "name": "Discord Social Credit Bot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "simple_health": "/health/simple", 
            "status": "/status",
            "profiles_stats": "/profiles/stats",
            "profiles_list": "/profiles",
            "user_profile": "/profiles/{user_id}",
            "simulate_message": "/simulate_message",
            "docs": "/docs"
        }
    }


@app.post("/simulate_message")
async def simulate_message(request: SimulateMessageRequest):
    """
    Simulate a Discord message for testing purposes.
    
    This creates a mock Discord message and publishes it to the signal bus
    to trigger the full processing pipeline (LLM classification, orchestration).
    """
    try:
        # Create mock Discord message using the Signal structure
        message = DiscordMessage(
            signal_id=f"sim_{int(datetime.now().timestamp() * 1000)}",
            source="discord",  
            created_at=datetime.now(),  
            author={
                "user_id": request.user_id,
                "username": request.username
            },
            context={
                "guild_id": request.guild_id,
                "channel_id": request.channel_id,
                "message_id": f"sim_{int(datetime.now().timestamp() * 1000)}"
            },
            content=request.content,
            # Additional Discord-specific fields
            guild_name="Test Guild",
            channel_name="test-channel"
        )
        
        # Get signal bus and publish message
        signal_bus = get_signal_bus()
        
        # Create the signal data structure that matches what discord_client publishes
        signal_data = {"discord_message": message.model_dump()}
        
        # Publish to signal bus with SIGNAL_INGESTED type
        await signal_bus.publish(
            signal_type=SignalType.SIGNAL_INGESTED,
            data=signal_data,
            source="simulation"
        )
        
        logger.info(f"📝 Simulated Discord message published: user={request.username}, content='{request.content[:50]}...'")
        
        return {
            "success": True,
            "message": "Message simulation published to signal bus",
            "simulated_message": {
                "signal_id": message.signal_id,
                "content": message.content,
                "user": request.username,
                "user_id": request.user_id,
                "timestamp": message.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to simulate message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate message: {str(e)}")


def main():
    """Main entry point for the application."""
    config = get_settings()
    
    # Configure uvicorn logging
    uvicorn_config = uvicorn.Config(
        "main:app",
        host=config.admin_api_host,
        port=config.admin_api_port,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
    
    server = uvicorn.Server(uvicorn_config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()