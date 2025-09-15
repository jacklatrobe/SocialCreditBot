"""
LLM Integration Package

This package provides Language Model integration capabilities for the Discord
Observer/Orchestrator system, including message classification, content analysis,
and response generation using OpenAI's GPT models.
"""

from .client import (
    OpenAILLMClient,
    ClassificationResult,
    ClassificationCategory,
    LLMConfig,
    get_llm_client,
    init_llm_client,
    classify_signal
)

__all__ = [
    'OpenAILLMClient',
    'ClassificationResult', 
    'ClassificationCategory',
    'LLMConfig',
    'get_llm_client',
    'init_llm_client',
    'classify_signal'
]