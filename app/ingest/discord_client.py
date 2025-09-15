"""
Discord Ingest Client Implementation

This module implements the DiscordIngestClient that manages Discord WebSocket connections,
handles privileged message intents, and processes incoming messages according to PRD section 6.
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


class DiscordIngestClient:
    """
    Discord WebSocket client for message ingestion.
    
    Manages Discord connections, processes incoming messages, and publishes
    signals to the internal bus for further processing by the observer/orchestrator.
    """
    
    def __init__(self, config=None):
        """Initialize the Discord client."""
        self.config = config or get_settings()
        self.client: Optional[discord.Client] = None
        self.signal_bus = None
        self.message_processor = get_message_processor()
        self.error_handler = get_discord_error_handler()
        
        # Track connection state
        self._connected = False
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        
        # Message filtering settings
        self.monitored_channels: Set[int] = set()
        self.ignored_users: Set[int] = set()
        self.bot_users: Set[int] = set()
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_filtered': 0,
            'signals_published': 0,
            'connection_time': None,
            'last_activity': None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the Discord client and signal bus connections.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🔧 Initializing Discord Ingest Client...")
            
            # Get signal bus instance
            self.signal_bus = await get_signal_bus()
            
            # Create Discord client with required intents
            intents = discord.Intents.default()
            intents.message_content = True  # Privileged intent for message content
            intents.guilds = True
            intents.guild_messages = True
            
            self.client = discord.Client(intents=intents)
            
            # Set up event handlers
            self._setup_event_handlers()
            
            logger.info("✅ Discord client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Discord client: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Set up Discord client event handlers."""
        
        @self.client.event
        async def on_ready():
            """Handle client ready event."""
            logger.info(f"🤖 Discord client connected as {self.client.user}")
            self._connected = True
            self._reconnect_attempts = 0
            self.stats['connection_time'] = datetime.now(timezone.utc)
            
            # Identify bot users to filter out
            await self._identify_bot_users()
            
            logger.info(f"📊 Monitoring {len(self.client.guilds)} guilds")
        
        @self.client.event
        async def on_disconnect():
            """Handle client disconnect event."""
            logger.warning("🔌 Discord client disconnected")
            self._connected = False
        
        @self.client.event
        async def on_resumed():
            """Handle client resume event."""
            logger.info("🔄 Discord client resumed connection")
            self._connected = True
        
        @self.client.event
        async def on_message(message):
            """Handle incoming Discord messages."""
            await self._process_message(message)
        
        @self.client.event
        async def on_message_edit(before, after):
            """Handle message edit events."""
            # Process edited messages as new signals
            await self._process_message(after, is_edit=True)
        
        @self.client.event
        async def on_error(event, *args, **kwargs):
            """Handle Discord client errors."""
            error = args[0] if args else None
            if error:
                # Use error handler for logging and metrics
                await self.error_handler._handle_error(
                    error, 
                    self.error_handler._classify_error(error), 
                    1
                )
            logger.error(f"❌ Discord client error in {event}: {args}")
    
    async def _identify_bot_users(self):
        """Identify bot users to filter out of processing."""
        try:
            # Add our own bot to ignored list
            if self.client.user:
                self.bot_users.add(self.client.user.id)
            
            # Scan guilds for other bot users
            for guild in self.client.guilds:
                async for member in guild.fetch_members(limit=None):
                    if member.bot:
                        self.bot_users.add(member.id)
            
            logger.info(f"🤖 Identified {len(self.bot_users)} bot users to filter")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not identify all bot users: {e}")
    
    async def _process_message(self, message: discord.Message, is_edit: bool = False):
        """
        Process an incoming Discord message using the message processor.
        
        Args:
            message: The Discord message object
            is_edit: Whether this is an edited message
        """
        try:
            # Update activity timestamp
            self.stats['last_activity'] = datetime.now(timezone.utc)
            self.stats['messages_processed'] += 1
            
            # Use message processor for advanced filtering and processing
            result = await self.message_processor.process_message(message, is_edit)
            
            if not result.should_process:
                self.stats['messages_filtered'] += 1
                logger.debug(f"🔍 Filtered message from {message.author}: {result.skip_reason}")
                return
            
            if result.signal:
                # Publish signal to bus with error handling
                async def _publish_signal():
                    return await self.signal_bus.publish(
                        signal_type=SignalType.SIGNAL_INGESTED,
                        data=result.signal.model_dump(),
                        source="discord_ingest"
                    )
                
                try:
                    success = await self.error_handler.handle_with_retry(_publish_signal)
                    if success:
                        self.stats['signals_published'] += 1
                        logger.debug(f"📨 Published signal for message from {message.author}")
                    else:
                        logger.warning(f"⚠️ Failed to publish signal for message {message.id}")
                except Exception as e:
                    logger.error(f"❌ Signal publishing failed after retries: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing message {message.id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        stats.update({
            'connected': self.is_connected(),
            'running': self.is_running(),
            'reconnect_attempts': self._reconnect_attempts,
            'monitored_channels': len(self.monitored_channels),
            'ignored_users': len(self.ignored_users),
            'bot_users': len(self.bot_users),
            'processor_stats': self.message_processor.get_stats(),
            'error_handler_stats': self.error_handler.get_metrics()
        })
        return stats
    
    async def start(self) -> bool:
        """
        Start the Discord client with error handling.
        
        Returns:
            bool: True if started successfully
        """
        async def _start_client():
            """Internal start method for error handling."""
            if not await self.initialize():
                raise RuntimeError("Failed to initialize Discord client")
            
            logger.info("🚀 Starting Discord Ingest Client...")
            self._running = True
            
            # Start the Discord client
            await self.client.start(self.config.discord_bot_token)
        
        try:
            # Use error handler for robust startup
            await self.error_handler.handle_with_retry(_start_client)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Discord client after retries: {e}")
            self._running = False
            return False
    
    async def stop(self):
        """Stop the Discord client gracefully."""
        logger.info("🛑 Stopping Discord Ingest Client...")
        self._running = False
        
        if self.client:
            await self.client.close()
        
        self._connected = False
        logger.info("✅ Discord client stopped")
    
    def is_connected(self) -> bool:
        """Check if the client is connected to Discord."""
        return self._connected and self.client and not self.client.is_closed()
    
    def is_running(self) -> bool:
        """Check if the client is running."""
        return self._running
    
    def add_monitored_channel(self, channel_id: int):
        """Add a channel to monitor for messages."""
        self.monitored_channels.add(channel_id)
        self.message_processor.add_monitored_channel(channel_id)
        logger.info(f"📢 Added channel {channel_id} to monitoring list")
    
    def remove_monitored_channel(self, channel_id: int):
        """Remove a channel from monitoring."""
        self.monitored_channels.discard(channel_id)
        self.message_processor.remove_monitored_channel(channel_id)
        logger.info(f"🔇 Removed channel {channel_id} from monitoring list")
    
    def add_ignored_user(self, user_id: int):
        """Add a user to the ignore list."""
        self.ignored_users.add(user_id)
        self.message_processor.add_blocked_user(user_id)
        logger.info(f"🚫 Added user {user_id} to ignore list")
    
    def remove_ignored_user(self, user_id: int):
        """Remove a user from the ignore list."""
        self.ignored_users.discard(user_id)
        self.message_processor.remove_blocked_user(user_id)
        logger.info(f"✅ Removed user {user_id} from ignore list")


# Global client instance
_discord_client: Optional[DiscordIngestClient] = None


async def get_discord_client() -> DiscordIngestClient:
    """Get the global Discord client instance."""
    global _discord_client
    if _discord_client is None:
        _discord_client = DiscordIngestClient()
    return _discord_client


async def init_discord_client() -> DiscordIngestClient:
    """Initialize and return the Discord client."""
    client = await get_discord_client()
    await client.initialize()
    return client