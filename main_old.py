#!/usr/bin/env python3
"""
Discord Observer/Orchestrator Bot - Main Entry Point

This is the main entry point for the Discord Social Credit Bot system.
It orchestrates the startup of all components in the correct order:

1. Signal Bus initialization
2. LLM Observer startup (classification service)
3. Message Orchestrator startup (response logic)
4. Discord Ingest Client startup (Discord connection)

The system follows Clean Code principles with proper error handling,
graceful shutdown, and comprehensive logging.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
from datetime import datetime

from app.config import get_settings
from app.infra.bus import init_signal_bus, close_signal_bus
from app.observer import start_llm_observer, stop_llm_observer
from app.orchestrator import start_orchestrator, stop_orchestrator
from app.ingest.discord_client import run_discord_bot, stop_discord_bot


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('discord_bot.log', encoding='utf-8')
    ]
)

# Set stdout encoding to handle Unicode on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

logger = logging.getLogger(__name__)


class DiscordBotSystem:
    """
    Main system coordinator following Single Responsibility Principle.
    
    Manages the lifecycle of all system components with proper error handling
    and graceful shutdown capabilities.
    """
    
    def __init__(self):
        """Initialize the bot system."""
        self.config = get_settings()
        self.discord_client: Optional[object] = None
        self.running = False
        self.start_time: Optional[datetime] = None
        
        # Track component states
        self.components_started = {
            'signal_bus': False,
            'llm_observer': False,
            'orchestrator': False,
            'discord_client': False
        }
    
    async def start(self) -> bool:
        """
        Start all system components in the correct order.
        
        Returns:
            bool: True if all components started successfully
        """
        try:
            logger.info("[STARTING] Discord Observer/Orchestrator Bot System")
            self.start_time = datetime.now()
            
            # 1. Start Signal Bus (core communication layer)
            logger.info("[SIGNAL_BUS] Starting Signal Bus...")
            signal_bus = await init_signal_bus()
            self.components_started['signal_bus'] = True
            logger.info("[SIGNAL_BUS] Signal Bus started")
            
            # 2. Start LLM Observer (message classification)
            logger.info("[LLM_OBSERVER] Starting LLM Observer...")
            await start_llm_observer()
            self.components_started['llm_observer'] = True
            logger.info("[LLM_OBSERVER] LLM Observer started")
            
            # 3. Start Message Orchestrator (response logic)
            logger.info("[ORCHESTRATOR] Starting Message Orchestrator...")
            await start_orchestrator()
            self.components_started['orchestrator'] = True
            logger.info("[ORCHESTRATOR] Message Orchestrator started")
            
            # 4. Start Discord Client (Discord connection)
            logger.info("[DISCORD] Starting Discord Client...")
            self.discord_client = await init_discord_client()
            
            # Connect Discord client to signal bus
            await self.discord_client.connect_signal_bus()
            
            if await self.discord_client.start():
                self.components_started['discord_client'] = True
                logger.info("[DISCORD] Discord Client connected successfully")
            else:
                raise RuntimeError("Failed to start Discord client")
            
            self.running = True
            uptime = datetime.now() - self.start_time
            logger.info(f"[SUCCESS] System fully operational! Startup took {uptime.total_seconds():.2f}s")
            
            # Display system status
            await self._log_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start system: {e}")
            await self._cleanup_partial_startup()
            return False
    
    async def stop(self):
        """Stop all components gracefully in reverse order."""
        logger.info("[SHUTDOWN] Shutting down Discord Bot System...")
        
        try:
            # Stop in reverse order of startup
            if self.components_started['discord_client'] and self.discord_client:
                logger.info("[DISCORD] Stopping Discord Client...")
                await self.discord_client.stop()
                logger.info("[DISCORD] Discord Client stopped")
            
            if self.components_started['orchestrator']:
                logger.info("[ORCHESTRATOR] Stopping Message Orchestrator...")
                await stop_orchestrator()
                logger.info("[ORCHESTRATOR] Message Orchestrator stopped")
            
            if self.components_started['llm_observer']:
                logger.info("[LLM_OBSERVER] Stopping LLM Observer...")
                await stop_llm_observer()
                logger.info("[LLM_OBSERVER] LLM Observer stopped")
            
            if self.components_started['signal_bus']:
                logger.info("[SIGNAL_BUS] Stopping Signal Bus...")
                await close_signal_bus()
                logger.info("[SIGNAL_BUS] Signal Bus stopped")
            
            self.running = False
            uptime = datetime.now() - self.start_time if self.start_time else None
            uptime_str = f" (uptime: {uptime})" if uptime else ""
            logger.info(f"[SUCCESS] System shutdown complete{uptime_str}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error during shutdown: {e}")
    
    async def _cleanup_partial_startup(self):
        """Clean up any components that started before failure."""
        logger.info("[CLEANUP] Cleaning up partial startup...")
        await self.stop()
    
    async def _log_system_status(self):
        """Log current system status and configuration."""
        config_status = {
            'Discord Token': 'CONFIGURED' if self.config.discord_bot_token else 'MISSING',
            'OpenAI API Key': 'CONFIGURED' if self.config.llm_api_key else 'MISSING',
            'LLM Model': self.config.llm_model,
            'Database Path': self.config.db_path,
            'Training Data': f"{'ENABLED' if self.config.training_data_enabled else 'DISABLED'} ({self.config.training_data_path})"
        }
        
        logger.info("[STATUS] System Configuration:")
        for key, value in config_status.items():
            logger.info(f"   {key}: {value}")
    
    async def run_forever(self):
        """Run the bot system until interrupted."""
        if not await self.start():
            return False
        
        try:
            logger.info("[RUNNING] Bot is running... Press Ctrl+C to stop")
            
            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("[INTERRUPT] Keyboard interrupt received")
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in main loop: {e}")
        finally:
            await self.stop()
        
        return True


# Global system instance
bot_system: Optional[DiscordBotSystem] = None


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"[SIGNAL] Received signal {signum}, initiating shutdown...")
        if bot_system and bot_system.running:
            # Create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.create_task(bot_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    global bot_system
    
    try:
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
        # Create and run the bot system
        bot_system = DiscordBotSystem()
        success = await bot_system.run_forever()
        
        if success:
            logger.info("[GOODBYE] Bot system exited successfully")
            return 0
        else:
            logger.error("[FAILURE] Bot system failed to start")
            return 1
            
    except Exception as e:
        logger.error(f"[FATAL] Fatal error in main: {e}")
        return 1


if __name__ == "__main__":
    """Entry point when run directly."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("[GOODBYE] Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FATAL] Fatal error: {e}")
        sys.exit(1)