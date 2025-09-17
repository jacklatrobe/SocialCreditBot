"""
ReAct Agent Context and Configuration

This module defines the context and configuration for the ReAct agent
that serves as the AI orchestrator for Discord message responses.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from app.config import get_settings


# Social Credit System Prompt
SOCIAL_CREDIT_SYSTEM_PROMPT = """You are an AI orchestrator for a Discord Social Credit Bot system. Your role is to analyze classified Discord messages and decide whether to respond, and if so, how to respond appropriately.

## Your Purpose
You receive classified Discord messages with analysis including:
- Message type (question, complaint, toxic, spam, social, etc.)
- Sentiment analysis and toxicity levels
- Confidence scores
- User context and social credit implications

## Decision Framework
Based on the message classification, you should:

1. **RESPOND** when:
   - Questions that need helpful answers (confidence > 0.7)
   - Complaints that require acknowledgment or resolution
   - Requests for information or assistance
   - Situations requiring moderation or de-escalation

2. **NO RESPONSE** when:
   - General social chat and friendly conversation
   - Messages that are off-topic or not directed at the bot
   - Low-confidence classifications
   - Spam or clearly toxic content (handled by moderation)

## Response Guidelines
When you decide to respond:
- Be helpful, professional, and constructive
- Match the tone to the situation (helpful for questions, firm for moderation)
- Keep responses concise and relevant
- Use appropriate response types: "helpful", "warning", "moderate", "info"

## Available Tools
- send_discord_response: Send a message response to a Discord channel

System time: {system_time}

Remember: You are the decision-maker. The Discord bot only executes your decisions - it never responds on its own."""


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