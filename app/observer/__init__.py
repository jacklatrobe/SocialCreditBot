"""
Observer package for the Discord Observer/Orchestrator bot.

This package contains the Observer component that bridges message ingestion
and orchestration by performing LLM-based classification of Discord messages.
"""

from .llm_observer import LLMObserver, llm_observer, start_llm_observer, stop_llm_observer

__all__ = [
    'LLMObserver',
    'llm_observer',
    'start_llm_observer',
    'stop_llm_observer'
]