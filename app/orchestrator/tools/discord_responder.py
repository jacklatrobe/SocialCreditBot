"""
Discord Responder Tool

This module implements the DiscordResponderTool that handles sending responses
back to Discord channels based on orchestration decisions. It provides:

1. Response generation based on message classification
2. Discord API integration for sending messages
3. Context-aware response routing (channel/thread/reply)
4. Action logging for audit and monitoring
5. Template-based response generation

The tool follows the principle of single responsibility - it only sends responses
and never listens to incoming messages (that's handled by the DiscordIngestClient).
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
import aiohttp
from dataclasses import dataclass

from app.config import get_settings
from app.signals import Signal as BaseSignal, DiscordMessage
from app.infra.db import get_database
from app.infra.models import ActionRecord


logger = logging.getLogger(__name__)


class ResponseMode(Enum):
    """Response modes for Discord messages."""
    REPLY = "reply"          # Reply to the original message
    THREAD = "thread"        # Create or respond in a thread
    CHANNEL = "channel"      # Send to the channel (not as reply)
    DM = "dm"               # Direct message to the user


class ResponseTemplate(Enum):
    """Available response templates."""
    HELPFUL_RESPONSE = "helpful_response"
    SOCIAL_RESPONSE = "social_response"
    PROBLEM_ACKNOWLEDGMENT = "problem_acknowledgment"
    ESCALATION_NOTICE = "escalation_notice"
    GENERIC_HELP = "generic_help"


@dataclass
class ResponseContext:
    """Context information for generating responses."""
    signal_id: str
    channel_id: str
    guild_id: Optional[str]
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None


class DiscordResponderTool:
    """
    Tool for sending responses back to Discord channels.
    
    This tool implements the "responder" side of the Discord integration,
    handling outbound communication only. It receives Signal objects with
    context information and sends appropriate responses via Discord's REST API.
    
    Key responsibilities:
    - Generate contextually appropriate responses
    - Route responses to correct Discord location (channel/thread/reply)
    - Log all actions for audit and monitoring
    - Handle Discord API errors and rate limits
    - Provide response templates for different message types
    """
    
    name = "discord.respond"
    
    def __init__(self):
        """Initialize the Discord Responder Tool."""
        self.settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = "https://discord.com/api/v10"
        self._headers = {
            "Authorization": f"Bot {self.settings.discord_bot_token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (SocialCreditBot, 1.0)"
        }
        
        # Response templates
        self._templates = self._load_response_templates()
        
        # Statistics tracking
        self._stats = {
            'responses_sent': 0,
            'errors': 0,
            'rate_limits_hit': 0,
            'start_time': datetime.now(timezone.utc)
        }
        
        logger.info("Discord Responder Tool initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._headers
            )
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for different message types."""
        return {
            ResponseTemplate.HELPFUL_RESPONSE.value: (
                "Thanks for reaching out! Here's what I can help with:\n"
                "{guidance}\n\n"
                "Could you share more details about:\n"
                "• What you're trying to achieve\n"
                "• What you've tried so far\n"
                "• Any error messages you're seeing\n\n"
                "This helps me provide more targeted assistance!"
            ),
            
            ResponseTemplate.PROBLEM_ACKNOWLEDGMENT.value: (
                "Thanks for the report! I'd like to help investigate this issue.\n\n"
                "To help me understand what's happening, could you provide:\n"
                "• Your environment (OS/version)\n"
                "• Steps to reproduce the issue\n"
                "• Expected vs actual behavior\n"
                "• Any error logs or screenshots\n\n"
                "I'll look into this once I have those details!"
            ),
            
            ResponseTemplate.SOCIAL_RESPONSE.value: (
                "Hello! Great to see you here! 👋\n\n"
                "Feel free to ask questions, share what you're working on, or "
                "just chat with the community. We're here to help each other out!"
            ),
            
            ResponseTemplate.ESCALATION_NOTICE.value: (
                "I've flagged your message for human review. "
                "A moderator will take a look and get back to you soon.\n\n"
                "In the meantime, you can also check our documentation or "
                "search previous discussions for similar topics."
            ),
            
            ResponseTemplate.GENERIC_HELP.value: (
                "I'm here to help! While I analyze your message, you might find these resources useful:\n"
                "• Check the pinned messages for common solutions\n"
                "• Search previous discussions in this channel\n"
                "• Visit our documentation for detailed guides\n\n"
                "Feel free to provide more context if you need specific assistance!"
            )
        }
    
    async def run(
        self, 
        signal: BaseSignal, 
        text: Optional[str] = None, 
        mode: ResponseMode = ResponseMode.REPLY,
        template: Optional[ResponseTemplate] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for sending Discord responses.
        
        Args:
            signal: The original signal containing context
            text: Custom response text (overrides template if provided)
            mode: How to send the response (reply, thread, channel, dm)
            template: Response template to use if no custom text provided
            
        Returns:
            Dictionary containing response details and Discord API response
        """
        await self._ensure_session()
        
        try:
            # Extract context from signal
            context = self._extract_context(signal)
            
            # Generate response text
            if text is None:
                text = await self._generate_response_text(signal, template)
            
            # Send the response
            response = await self._send_response(context, text, mode)
            
            # Log the action
            await self._log_action(signal.signal_id, context, text, response)
            
            self._stats['responses_sent'] += 1
            
            logger.info(f"Response sent for signal {signal.signal_id}")
            
            return {
                'success': True,
                'signal_id': signal.signal_id,
                'discord_message_id': response.get('id'),
                'mode': mode.value,
                'response': response
            }
            
        except Exception as e:
            logger.error(f"Error sending response for signal {signal.signal_id}: {e}")
            self._stats['errors'] += 1
            
            return {
                'success': False,
                'signal_id': signal.signal_id,
                'error': str(e),
                'mode': mode.value
            }
    
    def _extract_context(self, signal: BaseSignal) -> ResponseContext:
        """Extract response context from signal."""
        context_data = signal.context
        
        return ResponseContext(
            signal_id=signal.signal_id,
            channel_id=context_data.get('channel_id'),
            guild_id=context_data.get('guild_id'),
            thread_id=context_data.get('thread_id'),
            message_id=context_data.get('message_id'),
            reply_to_id=context_data.get('reply_to_id'),
            author_id=signal.author.get('user_id') if isinstance(signal.author, dict) else None,
            author_name=signal.author.get('username') if isinstance(signal.author, dict) else signal.author,
            classification=context_data.get('classification')
        )
    
    async def _generate_response_text(
        self, 
        signal: BaseSignal, 
        template: Optional[ResponseTemplate] = None
    ) -> str:
        """
        Generate response text based on signal classification and template.
        
        Args:
            signal: The signal to generate a response for
            template: Specific template to use
            
        Returns:
            Generated response text
        """
        # If no template specified, determine from classification
        if template is None:
            template = self._select_template(signal)
        
        # Get template text
        template_text = self._templates.get(template.value, self._templates[ResponseTemplate.GENERIC_HELP.value])
        
        # Apply context-specific customizations
        response_text = await self._customize_response(signal, template_text)
        
        return response_text
    
    def _select_template(self, signal: BaseSignal) -> ResponseTemplate:
        """Select appropriate template based on signal classification."""
        context = signal.context
        classification = context.get('classification', {})
        
        if isinstance(classification, str):
            # Handle case where classification is stored as string
            try:
                import json
                classification = json.loads(classification)
            except (json.JSONDecodeError, TypeError):
                classification = {}
        
        message_type = classification.get('message_type', 'other')
        intent = classification.get('intent', 'other')
        
        # Map message types to templates
        if intent == 'help_request' or message_type == 'question':
            return ResponseTemplate.HELPFUL_RESPONSE
        elif intent == 'problem_report' or message_type == 'complaint':
            return ResponseTemplate.PROBLEM_ACKNOWLEDGMENT
        elif message_type == 'social':
            return ResponseTemplate.SOCIAL_RESPONSE
        else:
            return ResponseTemplate.GENERIC_HELP
    
    async def _customize_response(self, signal: BaseSignal, template_text: str) -> str:
        """
        Customize response text with context-specific information.
        
        Args:
            signal: The signal being responded to
            template_text: Base template text
            
        Returns:
            Customized response text
        """
        # Basic customization - can be enhanced with more sophisticated logic
        customized_text = template_text
        
        # Add user name if available
        author_name = None
        if isinstance(signal.author, dict):
            author_name = signal.author.get('username')
        elif isinstance(signal.author, str):
            author_name = signal.author
        
        if author_name and '{user}' in customized_text:
            customized_text = customized_text.replace('{user}', author_name)
        
        # Add guidance placeholder (could be enhanced with AI-generated content)
        if '{guidance}' in customized_text:
            customized_text = customized_text.replace(
                '{guidance}', 
                "I've analyzed your message and I'm ready to help!"
            )
        
        return customized_text
    
    async def _send_response(
        self, 
        context: ResponseContext, 
        text: str, 
        mode: ResponseMode
    ) -> Dict[str, Any]:
        """
        Send response via Discord REST API.
        
        Args:
            context: Response context information
            text: Response text to send
            mode: How to send the response
            
        Returns:
            Discord API response
        """
        # Ensure text doesn't exceed Discord's limit
        if len(text) > 2000:
            text = text[:1997] + "..."
            logger.warning(f"Response text truncated for signal {context.signal_id}")
        
        # Prepare request payload
        payload = {
            "content": text,
            "allowed_mentions": {
                "parse": [],  # Don't mention everyone/here by default
                "users": [context.author_id] if context.author_id else [],
                "roles": [],
                "replied_user": True
            }
        }
        
        # Add reply reference if appropriate
        if mode == ResponseMode.REPLY and context.message_id:
            payload["message_reference"] = {
                "message_id": context.message_id,
                "channel_id": context.channel_id
            }
            if context.guild_id:
                payload["message_reference"]["guild_id"] = context.guild_id
        
        # Determine endpoint
        endpoint = f"/channels/{context.channel_id}/messages"
        url = f"{self._base_url}{endpoint}"
        
        # Send the request
        async with self._session.post(url, json=payload) as response:
            if response.status == 429:  # Rate limited
                self._stats['rate_limits_hit'] += 1
                retry_after = response.headers.get('Retry-After', 1)
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(float(retry_after))
                # Retry once
                async with self._session.post(url, json=payload) as retry_response:
                    retry_response.raise_for_status()
                    return await retry_response.json()
            else:
                response.raise_for_status()
                return await response.json()
    
    async def _log_action(
        self, 
        signal_id: str, 
        context: ResponseContext, 
        text: str, 
        response: Dict[str, Any]
    ) -> None:
        """
        Log the response action to the database.
        
        Args:
            signal_id: ID of the signal being responded to
            context: Response context
            text: Response text that was sent
            response: Discord API response
        """
        try:
            action_id = f"resp_{signal_id}_{int(datetime.now(timezone.utc).timestamp())}"
            
            request_data = {
                "channel_id": context.channel_id,
                "guild_id": context.guild_id,
                "thread_id": context.thread_id,
                "message_id": context.message_id,
                "text": text[:500],  # Truncate for storage
                "mode": "reply"  # Default for now
            }
            
            response_data = {
                "discord_message_id": response.get('id'),
                "status": "success",
                "timestamp": response.get('timestamp')
            }
            
            db = await get_database()
            async with db.get_session() as session:
                action_record = ActionRecord(
                    action_id=action_id,
                    signal_id=signal_id,
                    action=self.name,
                    request=request_data,
                    response=response_data,
                    status="success"
                )
                
                session.add(action_record)
                await session.commit()
                
                logger.debug(f"Action logged: {action_id}")
                
        except Exception as e:
            logger.error(f"Failed to log action for signal {signal_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get responder statistics."""
        stats = self._stats.copy()
        
        runtime = datetime.now(timezone.utc) - stats['start_time']
        stats['runtime_seconds'] = runtime.total_seconds()
        stats['responses_per_minute'] = (
            stats['responses_sent'] / (runtime.total_seconds() / 60) 
            if runtime.total_seconds() > 0 else 0
        )
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the responder tool."""
        health = {
            'tool_name': self.name,
            'session_healthy': self._session is not None and not self._session.closed,
            'stats': self.get_stats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Test Discord API connectivity
        try:
            await self._ensure_session()
            test_url = f"{self._base_url}/users/@me"
            async with self._session.get(test_url) as response:
                health['discord_api_accessible'] = response.status == 200
        except Exception as e:
            health['discord_api_accessible'] = False
            health['discord_api_error'] = str(e)
        
        health['healthy'] = (
            health['session_healthy'] and 
            health['discord_api_accessible']
        )
        
        return health


# Factory function for creating the tool
def create_discord_responder_tool() -> DiscordResponderTool:
    """Create and return a DiscordResponderTool instance."""
    return DiscordResponderTool()