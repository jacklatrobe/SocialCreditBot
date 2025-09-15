"""
Discord Message Processing Pipeline

This module provides advanced message processing, filtering, and content extraction
capabilities for Discord messages before they are converted to signals.
"""
import re
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import discord

from app.signals.discord import DiscordMessage


logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for message processing pipeline."""
    
    # Content filtering
    min_content_length: int = 3
    max_content_length: int = 4000
    filter_empty_messages: bool = True
    filter_system_messages: bool = True
    
    # User filtering
    filter_bots: bool = True
    filter_webhooks: bool = True
    allowed_bot_ids: Set[int] = None
    blocked_user_ids: Set[int] = None
    
    # Content analysis
    extract_urls: bool = True
    extract_mentions: bool = True
    extract_emojis: bool = True
    detect_spam_patterns: bool = True
    
    # Channel filtering
    monitored_channels: Set[int] = None
    ignored_channels: Set[int] = None
    channel_types: Set[discord.ChannelType] = None
    
    def __post_init__(self):
        if self.allowed_bot_ids is None:
            self.allowed_bot_ids = set()
        if self.blocked_user_ids is None:
            self.blocked_user_ids = set()
        if self.monitored_channels is None:
            self.monitored_channels = set()
        if self.ignored_channels is None:
            self.ignored_channels = set()
        if self.channel_types is None:
            self.channel_types = {discord.ChannelType.text}


@dataclass
class ProcessingResult:
    """Result of message processing."""
    
    should_process: bool
    signal: Optional[DiscordMessage] = None
    skip_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DiscordMessageProcessor:
    """
    Advanced Discord message processor with filtering and content analysis.
    
    Provides sophisticated message filtering, content extraction, and signal creation
    with configurable policies for different server and channel types.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the message processor."""
        self.config = config or ProcessingConfig()
        
        # Compile regex patterns for performance
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._mention_pattern = re.compile(r'<@!?(\d+)>')
        self._role_mention_pattern = re.compile(r'<@&(\d+)>')
        self._channel_mention_pattern = re.compile(r'<#(\d+)>')
        self._emoji_pattern = re.compile(r'<a?:\w+:\d+>')
        self._spam_patterns = [
            re.compile(r'(.)\1{4,}', re.IGNORECASE),  # Repeated characters
            re.compile(r'^[A-Z\s!]+$'),  # All caps
            re.compile(r'(.{1,10})\1{3,}', re.IGNORECASE),  # Repeated phrases
        ]
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_accepted': 0,
            'messages_filtered': 0,
            'filter_reasons': {},
            'content_analysis': {
                'urls_extracted': 0,
                'mentions_extracted': 0,
                'emojis_extracted': 0,
                'spam_detected': 0
            }
        }
    
    async def process_message(self, message: discord.Message, is_edit: bool = False) -> ProcessingResult:
        """
        Process a Discord message through the complete pipeline.
        
        Args:
            message: Discord message object
            is_edit: Whether this is an edited message
            
        Returns:
            ProcessingResult with processing outcome
        """
        self.stats['messages_processed'] += 1
        
        try:
            # Step 1: Basic filtering
            filter_result = self._apply_filters(message)
            if not filter_result.should_process:
                self.stats['messages_filtered'] += 1
                self._update_filter_stats(filter_result.skip_reason)
                return filter_result
            
            # Step 2: Content analysis
            content_metadata = await self._analyze_content(message)
            
            # Step 3: Create enhanced signal
            signal = await self._create_enhanced_signal(message, is_edit, content_metadata)
            
            if signal:
                self.stats['messages_accepted'] += 1
                return ProcessingResult(
                    should_process=True,
                    signal=signal,
                    metadata=content_metadata
                )
            else:
                self.stats['messages_filtered'] += 1
                self._update_filter_stats("signal_creation_failed")
                return ProcessingResult(
                    should_process=False,
                    skip_reason="signal_creation_failed"
                )
                
        except Exception as e:
            logger.error(f"❌ Error processing message {message.id}: {e}")
            self.stats['messages_filtered'] += 1
            self._update_filter_stats("processing_error")
            return ProcessingResult(
                should_process=False,
                skip_reason=f"processing_error: {str(e)}"
            )
    
    def _apply_filters(self, message: discord.Message) -> ProcessingResult:
        """Apply all configured filters to determine if message should be processed."""
        
        # Filter system messages
        if self.config.filter_system_messages and message.type != discord.MessageType.default:
            return ProcessingResult(False, skip_reason="system_message")
        
        # Filter bots (unless explicitly allowed)
        if self.config.filter_bots and message.author.bot:
            if message.author.id not in self.config.allowed_bot_ids:
                return ProcessingResult(False, skip_reason="bot_message")
        
        # Filter webhooks
        if self.config.filter_webhooks and message.webhook_id is not None:
            return ProcessingResult(False, skip_reason="webhook_message")
        
        # Filter blocked users
        if message.author.id in self.config.blocked_user_ids:
            return ProcessingResult(False, skip_reason="blocked_user")
        
        # Filter by channel type
        if message.channel.type not in self.config.channel_types:
            return ProcessingResult(False, skip_reason="unsupported_channel_type")
        
        # Filter by monitored channels (if specified)
        if self.config.monitored_channels and message.channel.id not in self.config.monitored_channels:
            return ProcessingResult(False, skip_reason="channel_not_monitored")
        
        # Filter ignored channels
        if message.channel.id in self.config.ignored_channels:
            return ProcessingResult(False, skip_reason="channel_ignored")
        
        # Filter by content length
        content_length = len(message.content.strip())
        if self.config.filter_empty_messages and content_length == 0 and not message.attachments:
            return ProcessingResult(False, skip_reason="empty_content")
        
        if content_length < self.config.min_content_length:
            return ProcessingResult(False, skip_reason="content_too_short")
        
        if content_length > self.config.max_content_length:
            return ProcessingResult(False, skip_reason="content_too_long")
        
        return ProcessingResult(True)
    
    async def _analyze_content(self, message: discord.Message) -> Dict[str, Any]:
        """Perform content analysis and extract metadata."""
        metadata = {
            'content_length': len(message.content),
            'word_count': len(message.content.split()) if message.content else 0,
            'has_attachments': len(message.attachments) > 0,
            'attachment_count': len(message.attachments),
            'embed_count': len(message.embeds),
            'created_at': message.created_at.isoformat(),
        }
        
        # Extract URLs
        if self.config.extract_urls:
            urls = self._url_pattern.findall(message.content)
            metadata['urls'] = urls
            metadata['url_count'] = len(urls)
            if urls:
                self.stats['content_analysis']['urls_extracted'] += len(urls)
        
        # Extract mentions
        if self.config.extract_mentions:
            user_mentions = [int(uid) for uid in self._mention_pattern.findall(message.content)]
            role_mentions = [int(rid) for rid in self._role_mention_pattern.findall(message.content)]
            channel_mentions = [int(cid) for cid in self._channel_mention_pattern.findall(message.content)]
            
            metadata['user_mentions'] = user_mentions
            metadata['role_mentions'] = role_mentions  
            metadata['channel_mentions'] = channel_mentions
            metadata['total_mentions'] = len(user_mentions) + len(role_mentions) + len(channel_mentions)
            
            if metadata['total_mentions'] > 0:
                self.stats['content_analysis']['mentions_extracted'] += metadata['total_mentions']
        
        # Extract emojis
        if self.config.extract_emojis:
            custom_emojis = self._emoji_pattern.findall(message.content)
            metadata['custom_emojis'] = custom_emojis
            metadata['custom_emoji_count'] = len(custom_emojis)
            if custom_emojis:
                self.stats['content_analysis']['emojis_extracted'] += len(custom_emojis)
        
        # Detect spam patterns
        if self.config.detect_spam_patterns:
            spam_indicators = []
            for pattern in self._spam_patterns:
                if pattern.search(message.content):
                    spam_indicators.append(pattern.pattern)
            
            metadata['spam_indicators'] = spam_indicators
            metadata['spam_score'] = len(spam_indicators)
            if spam_indicators:
                self.stats['content_analysis']['spam_detected'] += 1
        
        # Author analysis
        metadata['author_analysis'] = {
            'is_bot': message.author.bot,
            'account_age_days': (datetime.now(timezone.utc) - message.author.created_at).days,
            'has_avatar': message.author.avatar is not None,
            'display_name': message.author.display_name,
            'username': message.author.name
        }
        
        # Guild/channel context
        if message.guild:
            metadata['guild_context'] = {
                'member_count': message.guild.member_count,
                'guild_created_days_ago': (datetime.now(timezone.utc) - message.guild.created_at).days,
                'guild_features': list(message.guild.features)
            }
        
        return metadata
    
    async def _create_enhanced_signal(
        self, 
        message: discord.Message, 
        is_edit: bool, 
        content_metadata: Dict[str, Any]
    ) -> Optional[DiscordMessage]:
        """Create a DiscordMessage signal with enhanced metadata."""
        try:
            # Use the factory method from DiscordMessage
            signal = DiscordMessage.from_discord_message(message, source="discord_processor")
            
            # Add processing metadata
            signal.metadata.update({
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'is_edit': is_edit,
                'processing_version': '1.0.0'
            })
            
            # Merge content analysis metadata
            signal.metadata.update(content_metadata)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced signal: {e}")
            return None
    
    def _update_filter_stats(self, reason: str):
        """Update filter statistics."""
        if reason not in self.stats['filter_reasons']:
            self.stats['filter_reasons'][reason] = 0
        self.stats['filter_reasons'][reason] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        for key in self.stats:
            if isinstance(self.stats[key], dict):
                for subkey in self.stats[key]:
                    self.stats[key][subkey] = 0
            else:
                self.stats[key] = 0
    
    def update_config(self, **kwargs):
        """Update processing configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"🔧 Updated config: {key} = {value}")
            else:
                logger.warning(f"⚠️ Unknown config key: {key}")
    
    def add_monitored_channel(self, channel_id: int):
        """Add a channel to monitoring list."""
        self.config.monitored_channels.add(channel_id)
        logger.info(f"📢 Added channel {channel_id} to monitoring")
    
    def remove_monitored_channel(self, channel_id: int):
        """Remove a channel from monitoring list."""
        self.config.monitored_channels.discard(channel_id)
        logger.info(f"🔇 Removed channel {channel_id} from monitoring")
    
    def add_ignored_channel(self, channel_id: int):
        """Add a channel to ignore list."""
        self.config.ignored_channels.add(channel_id)
        logger.info(f"🚫 Added channel {channel_id} to ignore list")
    
    def remove_ignored_channel(self, channel_id: int):
        """Remove a channel from ignore list."""
        self.config.ignored_channels.discard(channel_id)
        logger.info(f"✅ Removed channel {channel_id} from ignore list")
    
    def add_blocked_user(self, user_id: int):
        """Add a user to block list."""
        self.config.blocked_user_ids.add(user_id)
        logger.info(f"🚫 Added user {user_id} to block list")
    
    def remove_blocked_user(self, user_id: int):
        """Remove a user from block list."""
        self.config.blocked_user_ids.discard(user_id)
        logger.info(f"✅ Removed user {user_id} from block list")


# Global processor instance
_message_processor: Optional[DiscordMessageProcessor] = None


def get_message_processor() -> DiscordMessageProcessor:
    """Get the global message processor instance."""
    global _message_processor
    if _message_processor is None:
        _message_processor = DiscordMessageProcessor()
    return _message_processor


def init_message_processor(config: Optional[ProcessingConfig] = None) -> DiscordMessageProcessor:
    """Initialize and return the message processor."""
    global _message_processor
    _message_processor = DiscordMessageProcessor(config)
    return _message_processor