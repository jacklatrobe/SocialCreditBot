"""
OpenAI LLM Integration Module

This module provides the core LLM integration using OpenAI's GPT-4o-mini model
for message classification and analysis. Implements async API client, prompt
engineering, response parsing, and comprehensive error handling.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import openai
from openai import AsyncOpenAI

from app.config import get_settings
from app.signals.base import Signal


logger = logging.getLogger(__name__)


class ClassificationCategory(Enum):
    """Categories for message classification."""
    PURPOSE = "purpose"
    SENTIMENT = "sentiment"
    URGENCY = "urgency"
    TOXICITY = "toxicity"
    INTENT = "intent"


@dataclass
class ClassificationResult:
    """Result of message classification."""
    purpose: str
    sentiment: str
    urgency: str
    toxicity: str
    intent: str
    confidence: float
    reasoning: Optional[str] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'purpose': self.purpose,
            'sentiment': self.sentiment,
            'urgency': self.urgency,
            'toxicity': self.toxicity,
            'intent': self.intent,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'processing_time': self.processing_time,
            'model_used': self.model_used
        }


@dataclass
class LLMConfig:
    """Configuration for LLM operations."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    timeout: float = 30.0
    max_retries: int = 3


class PromptTemplate:
    """Template for LLM prompts with structured output."""
    
    CLASSIFICATION_SYSTEM_PROMPT = """You are an expert message classifier for a Discord bot that observes and moderates community interactions.

Your task is to analyze Discord messages and classify them across multiple dimensions. Provide structured, consistent classifications that help determine appropriate responses.

Classification Dimensions:
1. PURPOSE: What is the main purpose of this message?
   - question, statement, request, complaint, suggestion, greeting, social, spam, other

2. SENTIMENT: What is the emotional tone?
   - positive, negative, neutral, mixed, unclear

3. URGENCY: How urgent or important is this message?
   - low, medium, high, critical

4. TOXICITY: Does this message contain harmful content?
   - none, mild, moderate, high, severe

5. INTENT: What does the author want to achieve?
   - information, help, discussion, attention, problem-solving, social-interaction, other

Always respond in JSON format with these exact fields:
{
    "purpose": "...",
    "sentiment": "...",
    "urgency": "...",
    "toxicity": "...",
    "intent": "...",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification"
}

Be consistent, objective, and err on the side of caution for toxicity classification."""

    CLASSIFICATION_USER_PROMPT = """Analyze this Discord message:

Author: {author}
Channel: {channel}
Content: "{content}"

Context:
- Guild: {guild}
- Has attachments: {has_attachments}
- Mentions: {mentions}
- Message length: {length} characters

Classify this message according to the system instructions."""

    @classmethod
    def format_classification_prompt(
        cls, 
        signal: Signal, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format the classification prompt for a signal."""
        context = context or {}
        
        # Extract relevant information from signal
        author = signal.author.get('username', 'Unknown')
        channel = signal.context.get('channel_id', 'Unknown')
        guild = signal.context.get('guild_id', 'DM')
        content = signal.content[:2000]  # Truncate for token limits
        
        # Additional context
        has_attachments = len(getattr(signal, 'attachments', [])) > 0
        mentions = len(getattr(signal, 'mentions', []))
        length = len(signal.content)
        
        return cls.CLASSIFICATION_USER_PROMPT.format(
            author=author,
            channel=channel,
            content=content,
            guild=guild,
            has_attachments=has_attachments,
            mentions=mentions,
            length=length
        )


class OpenAILLMClient:
    """
    Async OpenAI client for message classification and analysis.
    
    Provides structured message classification, content analysis, and response
    generation using OpenAI's GPT models with proper error handling and rate limiting.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the OpenAI LLM client."""
        self.settings = get_settings()
        self.config = config or LLMConfig()
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.settings.llm_api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'average_response_time': 0.0,
            'requests_by_type': {},
            'errors_by_type': {},
            'last_request_time': None
        }
    
    async def classify_message(self, signal: Signal) -> ClassificationResult:
        """
        Classify a message signal using the LLM.
        
        Args:
            signal: Signal object containing message data
            
        Returns:
            ClassificationResult with classification data
        """
        start_time = datetime.now()
        
        try:
            # Format the prompt
            user_prompt = PromptTemplate.format_classification_prompt(signal)
            
            # Make API request
            response = await self._make_chat_completion(
                system_prompt=PromptTemplate.CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                request_type="classification"
            )
            
            # Parse response
            result = self._parse_classification_response(response)
            
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.model_used = self.config.model
            
            # Update stats
            self._update_stats("classification", True, processing_time, response)
            
            logger.debug(f"✅ Classified message: {result.purpose} ({result.confidence:.2f})")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("classification", False, processing_time)
            logger.error(f"❌ Classification failed: {e}")
            
            # Return fallback classification
            return ClassificationResult(
                purpose="other",
                sentiment="neutral",
                urgency="low",
                toxicity="none",
                intent="other",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                processing_time=processing_time,
                model_used=self.config.model
            )
    
    async def _make_chat_completion(
        self, 
        system_prompt: str, 
        user_prompt: str,
        request_type: str = "general"
    ) -> Dict[str, Any]:
        """Make a chat completion request to OpenAI."""
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Extract response data
            result = {
                'content': response.choices[0].message.content,
                'finish_reason': response.choices[0].finish_reason,
                'usage': response.usage.model_dump() if response.usage else {},
                'model': response.model
            }
            
            return result
            
        except openai.RateLimitError as e:
            logger.warning(f"⏱️ Rate limit hit: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"🔐 Authentication failed: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"⏰ Request timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise
    
    def _parse_classification_response(self, response: Dict[str, Any]) -> ClassificationResult:
        """Parse the classification response from the LLM."""
        try:
            # Parse JSON content
            content = response['content']
            data = json.loads(content)
            
            # Validate required fields
            required_fields = ['purpose', 'sentiment', 'urgency', 'toxicity', 'intent', 'confidence']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create result object
            result = ClassificationResult(
                purpose=str(data['purpose']).lower(),
                sentiment=str(data['sentiment']).lower(),
                urgency=str(data['urgency']).lower(),
                toxicity=str(data['toxicity']).lower(),
                intent=str(data['intent']).lower(),
                confidence=float(data['confidence']),
                reasoning=data.get('reasoning', None)
            )
            
            # Validate confidence range
            result.confidence = max(0.0, min(1.0, result.confidence))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Invalid response format: {e}")
            raise ValueError(f"Invalid response format: {e}")
    
    def _update_stats(
        self, 
        request_type: str, 
        success: bool, 
        response_time: float,
        response: Optional[Dict[str, Any]] = None
    ):
        """Update client statistics."""
        self.stats['total_requests'] += 1
        self.stats['last_request_time'] = datetime.now().isoformat()
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # Update request type counters
        if request_type not in self.stats['requests_by_type']:
            self.stats['requests_by_type'][request_type] = 0
        self.stats['requests_by_type'][request_type] += 1
        
        # Update average response time
        total_successful = self.stats['successful_requests']
        if total_successful > 0:
            current_avg = self.stats['average_response_time']
            self.stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        
        # Update token usage
        if response and 'usage' in response:
            usage = response['usage']
            if 'total_tokens' in usage:
                self.stats['total_tokens_used'] += usage['total_tokens']
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the OpenAI API."""
        try:
            # Simple test request
            test_response = await self._make_chat_completion(
                system_prompt="You are a test assistant. Respond with exactly: {'status': 'ok'}",
                user_prompt="Test",
                request_type="health_check"
            )
            
            return {
                'status': 'healthy',
                'model': self.config.model,
                'response_time': 0.0,  # Would need timing
                'api_accessible': True
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model,
                'error': str(e),
                'api_accessible': False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset client statistics."""
        keys_to_reset = [
            'total_requests', 'successful_requests', 'failed_requests',
            'total_tokens_used', 'average_response_time'
        ]
        for key in keys_to_reset:
            self.stats[key] = 0
        
        self.stats['requests_by_type'] = {}
        self.stats['errors_by_type'] = {}
        self.stats['last_request_time'] = None
        
        logger.info("📊 LLM client stats reset")


# Global client instance
_llm_client: Optional[OpenAILLMClient] = None


async def get_llm_client() -> OpenAILLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAILLMClient()
    return _llm_client


async def init_llm_client(config: Optional[LLMConfig] = None) -> OpenAILLMClient:
    """Initialize and return the LLM client."""
    global _llm_client
    _llm_client = OpenAILLMClient(config)
    return _llm_client


async def classify_signal(signal: Signal) -> ClassificationResult:
    """Convenience function to classify a signal."""
    client = await get_llm_client()
    return await client.classify_message(signal)