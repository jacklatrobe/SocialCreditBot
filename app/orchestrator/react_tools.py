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
        
        This creates a signal to actually send the Discord message through
        the proper infrastructure.
        
        Args:
            message: Message content to send
            channel_id: Discord channel ID
            user_id: Discord user ID to respond to
            message_id: Optional original message ID
            response_type: Type of response
            
        Returns:
            String confirmation of the action taken
        """
        import asyncio
        from app.infra.bus import get_signal_bus
        from app.signals import DiscordMessage, ActionLog
        
        try:
            logger.info(f"🤖 ReAct agent TOOL EXECUTION START: send_discord_response")
            logger.info(f"   Target user: {user_id}")
            logger.info(f"   Target channel: {channel_id}")
            logger.info(f"   Response type: {response_type}")
            logger.info(f"   Message content: {repr(message)}")
            logger.info(f"   Reply to message: {message_id}")
            
            # Create an ActionLog signal to trigger Discord response
            action_signal = ActionLog(
                signal_id=f"discord_response_{int(__import__('time').time() * 1000)}",
                action_type="discord_response",
                target_id=user_id,
                metadata={
                    "response_content": message,
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "message_id": message_id,
                    "response_type": response_type,
                    "source": "react_agent_tool"
                }
            )
            
            # Send the signal through the bus
            try:
                bus = get_signal_bus()
                # Run the async publish in the current event loop or create one
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, create a task
                        task = loop.create_task(bus.publish(action_signal))
                        logger.info(f"📤 Discord response signal queued for async processing")
                    else:
                        # Run synchronously if no loop is running
                        loop.run_until_complete(bus.publish(action_signal))
                        logger.info(f"📤 Discord response signal published synchronously")
                except RuntimeError:
                    # Create new event loop if needed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(bus.publish(action_signal))
                    logger.info(f"📤 Discord response signal published in new event loop")
                    
            except Exception as signal_error:
                logger.error(f"Failed to publish Discord response signal: {signal_error}")
                # Fall back to just logging
                logger.warning(f"📝 FALLBACK: Would send Discord message to channel {channel_id}: {message[:100]}...")
            
            result = f"✅ Discord response queued for user {user_id} in channel {channel_id}: {message[:100]}..."
            logger.info(f"🤖 ReAct agent TOOL EXECUTION SUCCESS: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed to send Discord response: {str(e)}"
            logger.error(f"🤖 ReAct agent TOOL EXECUTION ERROR: {error_msg}")
            logger.error(f"   Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            return error_msg
    
    async def _arun(
        self,
        message: str,
        channel_id: str, 
        user_id: str,
        message_id: Optional[str] = None,
        response_type: str = "helpful",
        **kwargs: Any,
    ) -> str:
        """Async version with comprehensive logging."""
        logger.info(f"🤖 ReAct agent ASYNC TOOL EXECUTION: send_discord_response")
        return self._run(message, channel_id, user_id, message_id, response_type, **kwargs)


# List of available tools for the ReAct agent
ORCHESTRATOR_TOOLS = [DiscordResponseTool()]