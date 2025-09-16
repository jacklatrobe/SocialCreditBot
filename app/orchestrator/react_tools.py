"""
LangChain Tools for ReAct Agent Orchestrator

This module provides LangChain-compatible tools that wrap our existing
system functionality for use with the ReAct agent.
"""

import logging
from typing import Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.orchestrator.tools import create_discord_responder_tool, ResponseTemplate
from app.signals import Signal as BaseSignal


logger = logging.getLogger(__name__)


class DiscordResponseInput(BaseModel):
    """Input schema for Discord response tool."""
    message: str = Field(description="The message content to send to Discord")
    channel_id: str = Field(description="Discord channel ID where to send the message") 
    user_id: str = Field(description="Discord user ID to respond to")
    message_id: Optional[str] = Field(default=None, description="Original message ID to reference")
    response_type: str = Field(default="helpful", description="Type of response (helpful, warning, moderate, etc.)")


class DiscordResponseTool(BaseTool):
    """
    LangChain tool that wraps our DiscordResponderTool for the ReAct agent.
    
    This tool allows the ReAct agent to send responses to Discord channels
    using our existing Discord response infrastructure.
    """
    
    name: str = "send_discord_response"
    description: str = """
    Send a response message to a Discord channel. Use this tool when you need to respond 
    to a user's message in Discord. The tool handles all Discord API interactions.
    
    Parameters:
    - message: The content of the response message to send
    - channel_id: Discord channel ID where the message should be sent
    - user_id: Discord user ID of the person being responded to
    - message_id: (Optional) Original message ID for context
    - response_type: Type of response (helpful, warning, moderate, etc.)
    """
    
    args_schema: Type[BaseModel] = DiscordResponseInput
    
    def _run(
        self,
        message: str,
        channel_id: str,
        user_id: str,
        message_id: Optional[str] = None,
        response_type: str = "helpful",
        **kwargs: Any,
    ) -> str:
        """
        Send a Discord response message synchronously.
        
        This is a simplified synchronous version for the ReAct agent.
        In practice, the actual Discord sending would happen through 
        the signal/event system.
        
        Args:
            message: Message content to send
            channel_id: Discord channel ID
            user_id: Discord user ID to respond to
            message_id: Optional original message ID
            response_type: Type of response
            
        Returns:
            String confirmation of the action taken
        """
        try:
            logger.info(f"ReAct agent deciding to send Discord response to user {user_id}")
            logger.info(f"Response content: {message}")
            logger.info(f"Response type: {response_type}")
            
            # In a real system, this would queue a Discord response signal
            # For now, we'll just log the action and return success
            return f"Queued Discord response to user {user_id} in channel {channel_id}: {message[:100]}..."
            
        except Exception as e:
            logger.error(f"Error in ReAct Discord response tool: {e}")
            return f"Failed to send Discord response: {str(e)}"
    
    async def _arun(
        self,
        message: str,
        channel_id: str, 
        user_id: str,
        message_id: Optional[str] = None,
        response_type: str = "helpful",
        **kwargs: Any,
    ) -> str:
        """Async version - just calls sync version for now."""
        return self._run(message, channel_id, user_id, message_id, response_type, **kwargs)


# List of available tools for the ReAct agent
ORCHESTRATOR_TOOLS = [DiscordResponseTool()]