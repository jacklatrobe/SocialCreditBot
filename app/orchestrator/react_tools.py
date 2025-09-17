"""
LangChain Tools for ReAct Agent Orchestrator

This module provides LangChain-compatible tools that wrap our existing
system functionality for use with the ReAct agent.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Type
from datetime import datetime, timezone
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.orchestrator.tools import create_discord_responder_tool, ResponseTemplate
from app.signals import Signal as BaseSignal


logger = logging.getLogger(__name__)


# Global context store for current ReAct execution
_current_execution_context: Optional[Dict[str, Any]] = None


def set_execution_context(signal: BaseSignal, context: Dict[str, Any]) -> None:
    """Set the current execution context for tools to access."""
    global _current_execution_context
    _current_execution_context = {
        'signal': signal,
        'context': context
    }


def get_execution_context() -> Optional[Dict[str, Any]]:
    """Get the current execution context."""
    return _current_execution_context


def clear_execution_context() -> None:
    """Clear the execution context after processing."""
    global _current_execution_context
    _current_execution_context = None


class DiscordResponseInput(BaseModel):
    """Input schema for Discord response tool."""
    response_content: str = Field(description="The message content to send to Discord")


class DiscordResponseTool(BaseTool):
    """
    LangChain tool that wraps our DiscordResponderTool for the ReAct agent.
    
    This tool allows the ReAct agent to send responses to Discord channels.
    It automatically determines where to send the response based on the user's
    last message context from their profile.
    """
    
    name: str = "send_discord_response"
    description: str = """
    Send a response message to a Discord channel. Use this tool when you decide to respond 
    to a user's message. The tool automatically figures out where to send the response
    based on the user's most recent message.
    
    Parameters:
    - response_content: The content of the response message to send
    """
    
    args_schema: Type[BaseModel] = DiscordResponseInput
    
    async def _run(
        self,
        response_content: str,
        **kwargs: Any,
    ) -> str:
        """
        Send a Discord response message.
        
        This method gets the user context from the current ReAct state and automatically
        determines where to send the response based on their last message.
        
        Args:
            response_content: Message content to send
            
        Returns:
            String confirmation of the action taken
        """
        # Required imports for Discord message sending
        import asyncio
        
        try:
            logger.info(f"🤖 ReAct agent TOOL EXECUTION START: send_discord_response")
            logger.info(f"   Response content: {repr(response_content)}")
            
            # Get execution context
            exec_context = get_execution_context()
            
            if exec_context and exec_context.get('signal'):
                signal = exec_context['signal']
                signal_id = signal.signal_id
                logger.info(f"   Got signal context: {signal_id}")
                
                # Extract channel and user info from the signal context and author
                channel_id = signal.context.get('channel_id')
                user_id = signal.author.get('user_id')
                message_id = signal.context.get('message_id')
                
                if channel_id and user_id:
                    logger.info(f"   Target user: {user_id}")
                    logger.info(f"   Target channel: {channel_id}")
                    logger.info(f"   Reply to message: {message_id}")
                    
                    # Use the existing DiscordResponderTool to send the actual Discord message
                    from app.orchestrator.tools import create_discord_responder_tool, ResponseMode
                    discord_tool = create_discord_responder_tool()
                    
                    # Send the Discord message directly using the responder tool
                    result = await discord_tool.run(
                        signal=signal,
                        text=response_content,
                        mode=ResponseMode.REPLY  # Reply to the original message
                    )
                    
                    logger.info(f"📤 Discord message sent successfully")
                    logger.info(f"   Response ID: {result.get('response_id', 'unknown')}")
                    logger.info(f"   Discord API Response: {result.get('api_response', {}).get('id', 'unknown')}")
                    
                    return f"✅ Discord response sent: {response_content[:100]}..."
                    
                else:
                    error_msg = f"❌ Cannot send Discord response: missing channel_id ({channel_id}) or user_id ({user_id})"
                    logger.error(error_msg)
                    return error_msg
            else:
                error_msg = "❌ Cannot send Discord response: no execution context available"
                logger.error(error_msg)
                return error_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to send Discord response: {str(e)}"
            logger.error(f"🤖 ReAct agent TOOL EXECUTION ERROR: {error_msg}")
            logger.error(f"   Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            return error_msg
    
    async def _arun(
        self,
        response_content: str,
        **kwargs: Any,
    ) -> str:
        """Async version with comprehensive logging."""
        logger.info(f"🤖 ReAct agent ASYNC TOOL EXECUTION: send_discord_response")
        return await self._run(response_content, **kwargs)


# List of available tools for the ReAct agent
ORCHESTRATOR_TOOLS = [DiscordResponseTool()]