"""
Discord Ingest Client Implementation

This module implements a proper discord.py Bot that connects, listens, and stays running.
It follows discord.py best practices with event decorators and proper bot lifecycle.
"""
import asyncio
import logging
from typing import Optional, Set, List, Dict, Any
from datetime import datetime, timezone
import discord
from discord.ext import commands

from app.config import get_settings
from app.signals.discord import DiscordMessage
from app.signals.base import Signal
from app.infra.bus import get_signal_bus, SignalType
from app.ingest.message_processor import get_message_processor, ProcessingConfig
from app.ingest.error_handler import get_discord_error_handler, with_discord_retry


logger = logging.getLogger(__name__)


class SocialCreditBot(discord.Client):
    """
    Discord Bot following proper discord.py patterns.
    
    This bot connects, listens, and stays running using the standard discord.py Client pattern.
    It processes messages and publishes signals for the Social Credit system.
    """
    
    def __init__(self):
        # Configure intents for message content access
        intents = discord.Intents.default()
        intents.message_content = True  # Required for message content access
        intents.guilds = True
        intents.guild_messages = True
        
        super().__init__(intents=intents)
        
        # Initialize components
        self.config = get_settings()
        self.signal_bus = None
        self.message_processor = get_message_processor()
        self.error_handler = get_discord_error_handler()
        
        # Statistics tracking
        self.stats = {
            'messages_processed': 0,
            'messages_filtered': 0,
            'signals_published': 0,
            'connection_time': None,
            'last_activity': None
        }
        
        # Bot user tracking
        self.bot_users: Set[int] = set()
    
    async def setup_hook(self):
        """
        Setup hook called after login but before connecting to Discord gateway.
        Perfect for async initialization tasks.
        """
        try:
            logger.info("🔧 Setting up Discord bot...")
            
            # Connect to signal bus
            self.signal_bus = get_signal_bus()
            logger.info("✅ Connected to signal bus")
            
            logger.info("✅ Discord bot setup complete")
        except Exception as e:
            logger.error(f"❌ Failed to setup bot: {e}")
            raise
    
    async def on_ready(self):
        """Called when the bot has successfully connected to Discord."""
        logger.info(f'🤖 Discord bot logged in as {self.user}!')
        logger.info(f'📊 Monitoring {len(self.guilds)} guilds')
        
        # Update stats
        self.stats['connection_time'] = datetime.now(timezone.utc)
        
        # Identify bot users for filtering
        await self._identify_bot_users()
        
        logger.info("✅ Discord bot is ready and listening for messages!")
    
    async def on_message(self, message):
        """Handle incoming Discord messages."""
        # Log every message received for debugging
        logger.info(f"📨 Received message from {message.author} in #{message.channel}: '{message.content[:100]}...' (ID: {message.id})")
        
        # Don't process our own messages
        if message.author == self.user:
            logger.debug(f"🚫 Skipping own message: {message.id}")
            return
        
        await self._process_message(message)
    
    async def on_message_edit(self, before, after):
        """Handle message edit events."""
        # Don't process our own messages
        if after.author == self.user:
            return
        
        await self._process_message(after, is_edit=True)
    
    async def on_error(self, event, *args, **kwargs):
        """Handle bot errors."""
        error = args[0] if args else None
        logger.error(f"❌ Discord bot error in {event}: {error}")
        
        if error and self.error_handler:
            await self.error_handler._handle_error(
                error,
                self.error_handler._classify_error(error),
                1
            )
    
    async def on_disconnect(self):
        """Handle disconnect events."""
        logger.warning("🔌 Discord bot disconnected")
    
    async def on_resumed(self):
        """Handle resume events."""
        logger.info("🔄 Discord bot resumed connection")
    
    async def _identify_bot_users(self):
        """Identify bot users to filter out of processing."""
        try:
            # Add our own bot to ignored list
            if self.user:
                self.bot_users.add(self.user.id)
            
            # Scan guilds for other bot users
            for guild in self.guilds:
                async for member in guild.fetch_members(limit=None):
                    if member.bot:
                        self.bot_users.add(member.id)
            
            logger.info(f"🤖 Identified {len(self.bot_users)} bot users to filter")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not identify all bot users: {e}")
    
    async def _process_message(self, message: discord.Message, is_edit: bool = False):
        """Process an incoming Discord message."""
        logger.info(f"🔄 Processing message from {message.author} (ID: {message.id}, Edit: {is_edit})")
        
        try:
            # Update activity timestamp
            self.stats['last_activity'] = datetime.now(timezone.utc)
            self.stats['messages_processed'] += 1
            
            # Use message processor for filtering and processing
            result = await self.message_processor.process_message(message, is_edit)
            
            if not result.should_process:
                self.stats['messages_filtered'] += 1
                logger.debug(f"🔍 Filtered message from {message.author}: {result.skip_reason}")
                return
            
            if result.signal and self.signal_bus:
                # Publish signal to bus
                try:
                    success = await self.signal_bus.publish(
                        signal_type=SignalType.SIGNAL_INGESTED,
                        data={"discord_message": result.signal.model_dump()},
                        source="discord_ingest"
                    )
                    
                    if success:
                        self.stats['signals_published'] += 1
                        logger.debug(f"📨 Published signal for message from {message.author}")
                    else:
                        logger.warning(f"⚠️ Failed to publish signal for message {message.id}")
                        
                except Exception as e:
                    logger.error(f"❌ Signal publishing failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing message {message.id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        stats = self.stats.copy()
        stats.update({
            'connected': not self.is_closed(),
            'guild_count': len(self.guilds),
            'user_count': sum(guild.member_count for guild in self.guilds if guild.member_count),
            'bot_users': len(self.bot_users)
        })
        return stats


# Global bot instance
_discord_bot: Optional[SocialCreditBot] = None


def get_discord_bot() -> SocialCreditBot:
    """Get the global Discord bot instance."""
    global _discord_bot
    if _discord_bot is None:
        _discord_bot = SocialCreditBot()
    return _discord_bot


async def run_discord_bot():
    """
    Run the Discord bot following proper discord.py patterns.
    This function starts the bot and keeps it running.
    """
    bot = get_discord_bot()
    
    try:
        logger.info("🚀 Starting Discord bot...")
        # This is the proper way to run a discord.py bot
        # It handles the event loop and keeps the bot running
        await bot.start(bot.config.discord_bot_token)
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Discord bot error: {e}")
        raise
    finally:
        if not bot.is_closed():
            await bot.close()


async def stop_discord_bot():
    """Stop the Discord bot gracefully."""
    global _discord_bot
    if _discord_bot and not _discord_bot.is_closed():
        logger.info("� Stopping Discord bot...")
        await _discord_bot.close()
        _discord_bot = None