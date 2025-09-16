"""
Orchestrator package for the Discord Observer/Orchestrator bot.

This package contains the ReAct agent-based orchestration logic that coordinates between
message ingestion, classification, and response generation using AI decision making.
"""

from .core import (
    MessageOrchestrator,
    get_orchestrator,
    start_orchestrator,
    stop_orchestrator
)
__all__ = [
    'MessageOrchestrator',
    'get_orchestrator', 
    'start_orchestrator',
    'stop_orchestrator'
]