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


class KnowledgebaseInput(BaseModel):
    """Input schema for knowledgebase query tool."""
    query: str = Field(description="The search query to find relevant knowledge")


# Simple knowledgebase - replace these strings with your actual knowledge content
KNOWLEDGEBASE_ENTRIES = [
    "DayZ is a survival game where players must find food, water, and shelter while avoiding zombies and other players.",
    "DayZ servers reset loot spawns periodically, so check back at locations if you don't find items initially.",
    "In DayZ, always carry bandages or rags to stop bleeding from zombie attacks or player encounters.",
    "Social credit scores are earned by helping other players and being positive in chat.",
    "Social credit scores are reduced by toxic behavior, griefing, or spam messages.",
    "Users with low social credit scores may need additional guidance and support.",
    "The bot responds to questions, complaints, and situations requiring moderation.",
    "Users can ask about game mechanics, server rules, or general help topics.",
    "Response cooldowns prevent spam and give users time to read previous responses.",
    "High toxicity messages are handled by automated moderation systems.",
]


class KnowledgebaseTool(BaseTool):
    """
    Simple knowledgebase tool that searches through predefined knowledge entries.
    
    This tool allows the ReAct agent to query a knowledgebase for relevant information
    to help answer user questions or provide context for responses.
    """
    
    name: str = "knowledgebase"
    description: str = """
    Query the knowledgebase for relevant information. Use this tool to find knowledge
    about game mechanics, rules, social credit system, or other topics that might
    help you provide better responses to users.
    
    Parameters:
    - query: A search term or question to find relevant knowledge entries
    """
    
    args_schema: Type[BaseModel] = KnowledgebaseInput
    
    async def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Search the knowledgebase for relevant entries.
        
        Args:
            query: Search query string
            
        Returns:
            String containing matching knowledge entries
        """
        try:
            logger.info(f"🔍 Knowledgebase query: {repr(query)}")
            
            # Simple string matching - convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            matching_entries = []
            
            for entry in KNOWLEDGEBASE_ENTRIES:
                # Check if any word in the query appears in the entry
                if any(word in entry.lower() for word in query_lower.split()):
                    matching_entries.append(entry)
            
            if matching_entries:
                result = "Found relevant knowledge:\n" + "\n".join(f"• {entry}" for entry in matching_entries)
                logger.info(f"📚 Knowledgebase found {len(matching_entries)} entries for: {query}")
            else:
                result = f"No knowledge found for query: {query}"
                logger.info(f"📚 Knowledgebase found no entries for: {query}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Knowledgebase query failed: {str(e)}"
            logger.error(f"🔍 Knowledgebase ERROR: {error_msg}")
            return error_msg
    
    async def _arun(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Async version of knowledgebase query."""
        return await self._run(query, **kwargs)


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
ORCHESTRATOR_TOOLS = [DiscordResponseTool(), KnowledgebaseTool()]