#!/usr/bin/env python3
"""
Simple Discord Bot Test

This tests the proper discord.py bot implementation with minimal setup.
"""

import asyncio
import logging
import sys
import os

# Set up environment for testing
os.environ['DISCORD_BOT_TOKEN'] = os.getenv('DISCORD_BOT_TOKEN', 'your_token_here')
os.environ['LLM_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your_openai_key_here')

from app.ingest.discord_client import SocialCreditBot


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_bot():
    """Test the Discord bot connection."""
    if not os.getenv('DISCORD_BOT_TOKEN') or os.getenv('DISCORD_BOT_TOKEN') == 'your_token_here':
        logger.error("❌ Please set DISCORD_BOT_TOKEN environment variable!")
        return False
    
    logger.info("🧪 Testing Discord bot connection...")
    
    # Create bot instance
    bot = SocialCreditBot()
    
    try:
        # Test bot connection
        logger.info("🚀 Starting bot...")
        await bot.start(os.getenv('DISCORD_BOT_TOKEN'))
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        return False
    finally:
        if not bot.is_closed():
            await bot.close()
    
    return True


def main():
    """Run the bot test."""
    try:
        asyncio.run(test_bot())
    except KeyboardInterrupt:
        logger.info("🛑 Test stopped by user")


if __name__ == "__main__":
    main()