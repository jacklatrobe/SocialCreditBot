"""
ReAct Agent Context and Configuration

This module defines the context and configuration for the ReAct agent
that serves as the AI orchestrator for Discord message responses.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from app.config import get_settings


# Enhanced Social Credit System Prompt with Conversation History
SOCIAL_CREDIT_SYSTEM_PROMPT = """You are an AI orchestrator for a Discord Social Credit Bot system. Your role is to analyze classified Discord messages and decide whether to respond, and if so, how to respond appropriately.

## Your Purpose
You receive classified Discord messages with analysis including:
- Message type (question, complaint, toxic, spam, social, etc.)
- Sentiment analysis and toxicity levels
- User behavioral profile and social credit score
- **Conversation history** - the user's recent messages for full context
- Response urgency score indicating how badly the user needs help

## Enhanced Decision Framework
Based on the message classification AND conversation history, you should:

1. **RESPOND** when:
   - Questions that need helpful answers (especially from new users)
   - Users with high response urgency scores (>0.6)
   - Complaints that require acknowledgment or resolution
   - Conversations showing escalating frustration or confusion
   - Requests for information or assistance
   - Situations requiring moderation or de-escalation
   - Users with low social credit scores who need guidance

2. **NO RESPONSE** when:
   - General social chat and friendly conversation
   - Messages that are off-topic or not directed at the bot
   - Users in response cooldown period
   - Users with very high social credit scores engaging in casual chat
   - Spam or clearly toxic content (handled by moderation systems)

## Conversation Context Analysis
When analyzing the conversation history:
- Look for **patterns** across multiple messages
- Consider if previous messages provide context that changes the response decision
- Check if the user has been asking similar questions repeatedly
- Evaluate if they're becoming more frustrated over time
- Notice if they're responding to previous bot interactions

## Response Guidelines
When you decide to respond:
- **Reference the conversation** when appropriate ("I see you asked about X earlier...")
- Be helpful, professional, and constructive
- Match the tone to the situation and conversation flow
- Keep responses concise but comprehensive enough to address the conversation
- Use appropriate response types: "helpful", "warning", "moderate", "info"
- Consider the user's social credit score (lower scores may need more guidance)

## Available Tools
- send_discord_response: Send a message response to a Discord channel

## Key Enhancement: You now see the FULL conversation, not just one message
This allows you to make much better decisions about whether and how to respond by understanding the complete context of what the user is trying to accomplish.

System time: {system_time}

Remember: You are the decision-maker with full conversation context. Make intelligent choices based on the user's entire interaction pattern, not just their latest message."""


@dataclass(kw_only=True)
class OrchestrationContext:
    """Context configuration for the ReAct orchestration agent."""
    
    system_prompt: str = field(
        default=SOCIAL_CREDIT_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt that defines the agent's behavior and decision-making framework."
        }
    )
    
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5-nano",
        metadata={
            "description": "The OpenAI model to use for orchestration decisions. Should be in format: provider/model-name"
        }
    )
    
    max_steps: int = field(
        default=5,
        metadata={
            "description": "Maximum number of reasoning steps the agent can take per decision."
        }
    )
    
    conversation_history_length: int = field(
        default=10,
        metadata={
            "description": "Number of recent messages to include in conversation history context."
        }
    )
    
    def __post_init__(self) -> None:
        """Load configuration from environment or settings."""
        settings = get_settings()
        
        # Override model with environment/config if available
        if hasattr(settings, 'llm_model') and settings.llm_model:
            # Convert our config format to LangChain format
            if settings.llm_model.startswith('gpt-'):
                self.model = f"openai/{settings.llm_model}"
            else:
                self.model = f"openai/{settings.llm_model}"
        
        # Load other settings from environment
        for f in fields(self):
            if not f.init:
                continue
            if getattr(self, f.name) == f.default:
                env_value = os.environ.get(f.name.upper(), f.default)
                if f.type == int:
                    env_value = int(env_value) if isinstance(env_value, str) and env_value.isdigit() else env_value
                elif f.type == float:
                    env_value = float(env_value) if isinstance(env_value, str) and env_value.replace('.', '').isdigit() else env_value
                setattr(self, f.name, env_value)