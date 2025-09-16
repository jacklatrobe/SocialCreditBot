#!/usr/bin/env python3
"""
Discord Observer/Orchestrator Bot - Main Entry Point

This Discord Social Credit Bot follows proper discord.py patterns with:
- SocialCreditBot class extending discord.Client
- Proper event decorators (@bot.event)  
- Correct bot.run() usage for persistent connection
- Clean shutdown handling

The bot connects, listens, and stays running using standard discord.py patterns.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from app.config import get_settings
from app.infra.bus import init_signal_bus, close_signal_bus
from app.observer import start_llm_observer, stop_llm_observer
from app.orchestrator import start_orchestrator, stop_orchestrator
from app.ingest.discord_client import get_discord_bot


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('discord_bot.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


async def setup_system():
    """Initialize all system components in the correct order."""
    logger.info("🚀 Setting up Discord Social Credit Bot...")
    
    try:
        # 1. Initialize Signal Bus
        logger.info("📡 Initializing Signal Bus...")
        await init_signal_bus()
        logger.info("✅ Signal Bus ready")
        
        # 2. Start LLM Observer
        logger.info("🧠 Starting LLM Observer...")
        await start_llm_observer()
        logger.info("✅ LLM Observer ready")
        
        # 3. Start Message Orchestrator
        logger.info("🎭 Starting Message Orchestrator...")
        await start_orchestrator()
        logger.info("✅ Message Orchestrator ready")
        
        logger.info("✅ All system components ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup system: {e}")
        return False


async def shutdown_system():
    """Shutdown all system components gracefully."""
    logger.info("🛑 Shutting down system...")
    
    try:
        await stop_orchestrator()
        await stop_llm_observer()
        await close_signal_bus()
        logger.info("✅ System shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


async def main():
    """
    Main entry point using proper discord.py patterns.
    
    This function:
    1. Sets up all system components
    2. Gets the Discord bot instance
    3. Runs the bot using bot.run() (proper discord.py pattern)
    4. Handles graceful shutdown
    """
    config = get_settings()
    
    # Validate configuration
    if not config.discord_bot_token:
        logger.error("❌ DISCORD_BOT_TOKEN not set in environment!")
        return False
    
    if not config.llm_api_key:
        logger.error("❌ LLM_API_KEY (OpenAI) not set in environment!")
        return False
    
    logger.info("🤖 Starting Discord Social Credit Bot")
    logger.info(f"📊 Configuration: LLM={config.llm_model}, Training Data={'ON' if config.training_data_enabled else 'OFF'}")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Setup system components
        if not await setup_system():
            logger.error("❌ Failed to setup system components")
            return False
        
        # Get the Discord bot instance
        bot = get_discord_bot()
        
        logger.info("🚀 Starting Discord bot (this will block and keep running)...")
        
        # This is the proper discord.py way - it runs the bot and blocks
        # The bot will stay connected and listen for messages
        await bot.start(config.discord_bot_token)
        
    except KeyboardInterrupt:
        logger.info("🛑 Received Ctrl+C, shutting down...")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
    finally:
        await shutdown_system()
    
    return True


def run():
    """Synchronous entry point for the bot."""
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)