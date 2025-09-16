"""
State management for the ReAct Orchestration Agent

This module defines the state structures used by the LangGraph ReAct agent
for Discord message orchestration decisions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

from app.signals import Signal as BaseSignal


@dataclass 
class OrchestrationInput:
    """
    Input state for the ReAct orchestration agent.
    
    This represents the interface from the outside world - the classified
    Discord message that needs orchestration.
    """
    
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages for the ReAct conversation flow.
    Typically contains:
    1. HumanMessage with the classified message summary
    2. AIMessage with tool calls for orchestration decisions
    3. ToolMessage with results from Discord response actions
    4. AIMessage with final decision/reasoning
    """
    
    signal: Optional[BaseSignal] = field(default=None)
    """The original classified signal from the LLM Observer."""
    
    classification: Dict[str, Any] = field(default_factory=dict)
    """Classification results from the LLM Observer."""
    
    context: Dict[str, Any] = field(default_factory=dict)
    """Additional context about the Discord message and user."""


@dataclass
class OrchestrationState(OrchestrationInput):
    """
    Complete state for the ReAct orchestration agent.
    
    Extends the input state with additional runtime information.
    """
    
    is_last_step: IsLastStep = field(default=False)
    """
    Managed by LangGraph - indicates if we're at the max step limit.
    Prevents infinite loops in the ReAct cycle.
    """
    
    decision_made: bool = field(default=False)
    """Whether the agent has made a final orchestration decision."""
    
    action_taken: Optional[str] = field(default=None)
    """What action was taken (e.g., 'response_sent', 'no_action', 'escalated')."""
    
    response_sent: bool = field(default=False)
    """Whether a Discord response was successfully sent."""
    
    # Additional state can be added here as needed:
    # user_context: Dict[str, Any] = field(default_factory=dict)
    # social_credit_score: Optional[float] = field(default=None)
    # escalation_required: bool = field(default=False)