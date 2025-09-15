"""
Orchestrator package for the Discord Observer/Orchestrator bot.

This package contains the core orchestration logic that coordinates between
message ingestion, classification, and response generation.
"""

from .core import (
    MessageOrchestrator, 
    OrchestrationRule, 
    ActionType, 
    ResponsePriority,
    message_orchestrator,
    start_orchestrator,
    stop_orchestrator
)

__all__ = [
    'MessageOrchestrator',
    'OrchestrationRule',
    'ActionType',
    'ResponsePriority',
    'message_orchestrator',
    'start_orchestrator',
    'stop_orchestrator'
]