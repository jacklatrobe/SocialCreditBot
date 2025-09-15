"""
Message Classification System

This module provides specialized message classification capabilities with
enhanced prompt templates, classification rules, and analysis for different
Discord message types and contexts.
"""
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re

from app.llm.client import OpenAILLMClient, ClassificationResult, LLMConfig
from app.signals.base import Signal
from app.signals.discord import DiscordMessage


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of Discord messages for specialized classification."""
    REGULAR = "regular"
    QUESTION = "question"
    COMPLAINT = "complaint"
    SUPPORT_REQUEST = "support_request"
    SOCIAL = "social"
    ANNOUNCEMENT = "announcement"
    SPAM = "spam"
    TOXIC = "toxic"


class PurposeCategory(Enum):
    """Expanded purpose categories for classification."""
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    COMPLAINT = "complaint"
    SUGGESTION = "suggestion"
    GREETING = "greeting"
    SOCIAL = "social"
    ANNOUNCEMENT = "announcement"
    SPAM = "spam"
    OTHER = "other"


class SentimentLevel(Enum):
    """Sentiment analysis levels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class UrgencyLevel(Enum):
    """Urgency classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToxicityLevel(Enum):
    """Toxicity classification levels."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class ClassificationRules:
    """Rules for automatic classification based on patterns."""
    
    # Question indicators
    question_patterns: List[str] = field(default_factory=lambda: [
        r'\?',  # Contains question mark
        r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)',
        r'(help|assist|support|explain|tell me|show me)'
    ])
    
    # Complaint indicators  
    complaint_patterns: List[str] = field(default_factory=lambda: [
        r'(broken|bug|error|issue|problem|wrong|fail|not work)',
        r'(hate|dislike|terrible|awful|worst)',
        r'(frustrated|angry|annoyed)'
    ])
    
    # Spam indicators
    spam_patterns: List[str] = field(default_factory=lambda: [
        r'(.)\1{10,}',  # Repeated characters
        r'^[A-Z\s!]{20,}$',  # All caps long messages
        r'(click here|free|win|prize|offer)',
        r'(buy now|limited time|act fast)'
    ])
    
    # Toxic content indicators
    toxic_patterns: List[str] = field(default_factory=lambda: [
        r'(kill yourself|kys)',
        r'(retard|autist|cancer)',
        r'(f[*]?ck|sh[*]?t|damn|hell)',  # Mild profanity
    ])
    
    # Social/greeting indicators
    social_patterns: List[str] = field(default_factory=lambda: [
        r'^(hi|hello|hey|good morning|good evening|welcome)',
        r'(how are you|what\'s up|how\'s it going)',
        r'(goodbye|bye|see you|talk later|gn|good night)'
    ])
    
    def __post_init__(self):
        """Compile regex patterns for performance."""
        self.compiled_patterns = {
            'question': [re.compile(p, re.IGNORECASE) for p in self.question_patterns],
            'complaint': [re.compile(p, re.IGNORECASE) for p in self.complaint_patterns],
            'spam': [re.compile(p, re.IGNORECASE) for p in self.spam_patterns],
            'toxic': [re.compile(p, re.IGNORECASE) for p in self.toxic_patterns],
            'social': [re.compile(p, re.IGNORECASE) for p in self.social_patterns]
        }


@dataclass
class EnhancedClassificationResult(ClassificationResult):
    """Enhanced classification result with additional metadata."""
    
    message_type: Optional[MessageType] = None
    rule_matches: Dict[str, List[str]] = field(default_factory=dict)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'message_type': self.message_type.value if self.message_type else None,
            'rule_matches': self.rule_matches,
            'context_factors': self.context_factors,
            'content_analysis': self.content_analysis
        })
        return base_dict


class MessageClassificationService:
    """
    Advanced message classification service with rule-based pre-processing
    and LLM-based analysis for comprehensive message understanding.
    """
    
    def __init__(self, llm_client: Optional[OpenAILLMClient] = None):
        """Initialize the classification service."""
        self.llm_client = llm_client
        self.rules = ClassificationRules()
        
        # Classification statistics
        self.stats = {
            'total_classifications': 0,
            'rule_based_classifications': 0,
            'llm_classifications': 0,
            'classification_by_type': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'average_processing_time': 0.0
        }
    
    async def classify_message(self, signal: Signal) -> EnhancedClassificationResult:
        """
        Perform comprehensive message classification.
        
        Args:
            signal: Signal containing message to classify
            
        Returns:
            EnhancedClassificationResult with detailed classification
        """
        self.stats['total_classifications'] += 1
        
        try:
            # Step 1: Rule-based pre-classification
            rule_result = self._apply_classification_rules(signal)
            
            # Step 2: Content analysis
            content_analysis = self._analyze_content(signal)
            
            # Step 3: Context analysis
            context_factors = self._analyze_context(signal)
            
            # Step 4: LLM classification (if available)
            llm_result = None
            if self.llm_client:
                try:
                    llm_result = await self.llm_client.classify_message(signal)
                    self.stats['llm_classifications'] += 1
                except Exception as e:
                    logger.warning(f"⚠️ LLM classification failed, using rules: {e}")
            
            # Step 5: Combine results
            final_result = self._combine_classifications(
                signal, rule_result, llm_result, content_analysis, context_factors
            )
            
            # Update statistics
            self._update_stats(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            # Return fallback classification
            return self._create_fallback_classification(signal)
    
    def _apply_classification_rules(self, signal: Signal) -> Dict[str, Any]:
        """Apply rule-based classification patterns."""
        content = signal.content.lower()
        matches = {}
        
        # Check each pattern category
        for category, patterns in self.rules.compiled_patterns.items():
            category_matches = []
            for pattern in patterns:
                if pattern.search(content):
                    category_matches.append(pattern.pattern)
            
            if category_matches:
                matches[category] = category_matches
        
        # Determine primary classification based on rules
        if matches.get('toxic'):
            purpose = PurposeCategory.OTHER.value
            sentiment = SentimentLevel.NEGATIVE.value
            toxicity = ToxicityLevel.HIGH.value
            urgency = UrgencyLevel.MEDIUM.value
        elif matches.get('spam'):
            purpose = PurposeCategory.SPAM.value
            sentiment = SentimentLevel.NEUTRAL.value
            toxicity = ToxicityLevel.NONE.value
            urgency = UrgencyLevel.LOW.value
        elif matches.get('question'):
            purpose = PurposeCategory.QUESTION.value
            sentiment = SentimentLevel.NEUTRAL.value
            toxicity = ToxicityLevel.NONE.value
            urgency = UrgencyLevel.MEDIUM.value
        elif matches.get('complaint'):
            purpose = PurposeCategory.COMPLAINT.value
            sentiment = SentimentLevel.NEGATIVE.value
            toxicity = ToxicityLevel.NONE.value
            urgency = UrgencyLevel.HIGH.value
        elif matches.get('social'):
            purpose = PurposeCategory.SOCIAL.value
            sentiment = SentimentLevel.POSITIVE.value
            toxicity = ToxicityLevel.NONE.value
            urgency = UrgencyLevel.LOW.value
        else:
            purpose = PurposeCategory.STATEMENT.value
            sentiment = SentimentLevel.NEUTRAL.value
            toxicity = ToxicityLevel.NONE.value
            urgency = UrgencyLevel.LOW.value
        
        self.stats['rule_based_classifications'] += 1
        
        return {
            'purpose': purpose,
            'sentiment': sentiment,
            'urgency': urgency,
            'toxicity': toxicity,
            'intent': 'other',
            'confidence': 0.7 if matches else 0.3,
            'rule_matches': matches
        }
    
    def _analyze_content(self, signal: Signal) -> Dict[str, Any]:
        """Analyze message content for classification hints."""
        content = signal.content
        
        analysis = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'exclamation_count': content.count('!'),
            'question_count': content.count('?'),
            'caps_ratio': sum(1 for c in content if c.isupper()) / max(1, len(content)),
            'has_urls': 'http' in content.lower(),
            'has_mentions': '@' in content,
            'has_emojis': any(ord(c) > 127 for c in content)  # Basic emoji detection
        }
        
        # Content-based indicators
        if analysis['caps_ratio'] > 0.7 and analysis['length'] > 20:
            analysis['likely_spam'] = True
        
        if analysis['exclamation_count'] > 3:
            analysis['high_emotion'] = True
        
        if analysis['question_count'] > 0:
            analysis['has_questions'] = True
        
        return analysis
    
    def _analyze_context(self, signal: Signal) -> Dict[str, Any]:
        """Analyze message context for classification hints."""
        context = {}
        
        # Discord-specific context
        if isinstance(signal, DiscordMessage):
            context.update({
                'has_attachments': len(signal.attachments) > 0,
                'has_embeds': len(signal.embeds) > 0,
                'mention_count': len(signal.mentions),
                'is_dm': signal.context.get('guild_id') is None,
                'channel_name': signal.channel_name
            })
            
            # Channel-based hints
            if signal.channel_name:
                channel_lower = signal.channel_name.lower()
                if any(word in channel_lower for word in ['help', 'support', 'question']):
                    context['support_channel'] = True
                elif any(word in channel_lower for word in ['general', 'chat', 'random']):
                    context['social_channel'] = True
                elif any(word in channel_lower for word in ['announce', 'news', 'update']):
                    context['announcement_channel'] = True
        
        # Author context
        author_name = signal.author.get('username', '').lower()
        if any(word in author_name for word in ['bot', 'automated', 'system']):
            context['likely_bot'] = True
        
        return context
    
    def _combine_classifications(
        self,
        signal: Signal,
        rule_result: Dict[str, Any],
        llm_result: Optional[ClassificationResult],
        content_analysis: Dict[str, Any],
        context_factors: Dict[str, Any]
    ) -> EnhancedClassificationResult:
        """Combine rule-based and LLM classifications."""
        
        # Start with rule-based result
        combined = {
            'purpose': rule_result['purpose'],
            'sentiment': rule_result['sentiment'],
            'urgency': rule_result['urgency'],
            'toxicity': rule_result['toxicity'],
            'intent': rule_result['intent'],
            'confidence': rule_result['confidence'],
            'reasoning': 'Rule-based classification'
        }
        
        # Enhance with LLM result if available
        if llm_result and llm_result.confidence > 0.6:
            # Use LLM result if it has higher confidence
            if llm_result.confidence > combined['confidence']:
                combined.update({
                    'purpose': llm_result.purpose,
                    'sentiment': llm_result.sentiment,
                    'urgency': llm_result.urgency,
                    'toxicity': llm_result.toxicity,
                    'intent': llm_result.intent,
                    'confidence': llm_result.confidence,
                    'reasoning': f"LLM classification (was: {combined['reasoning']})"
                })
            else:
                # Combine confidences
                combined['confidence'] = (combined['confidence'] + llm_result.confidence) / 2
                combined['reasoning'] = 'Combined rule-based and LLM classification'
        
        # Apply context adjustments
        if context_factors.get('support_channel') and combined['purpose'] in ['statement', 'other']:
            combined['purpose'] = 'question'
            combined['intent'] = 'help'
        
        if context_factors.get('likely_bot'):
            combined['confidence'] *= 0.8  # Lower confidence for bot messages
        
        # Determine message type
        message_type = self._determine_message_type(combined, content_analysis, context_factors)
        
        return EnhancedClassificationResult(
            purpose=combined['purpose'],
            sentiment=combined['sentiment'],
            urgency=combined['urgency'],
            toxicity=combined['toxicity'],
            intent=combined['intent'],
            confidence=combined['confidence'],
            reasoning=combined['reasoning'],
            processing_time=llm_result.processing_time if llm_result else 0.0,
            model_used=llm_result.model_used if llm_result else 'rule-based',
            message_type=message_type,
            rule_matches=rule_result.get('rule_matches', {}),
            context_factors=context_factors,
            content_analysis=content_analysis
        )
    
    def _determine_message_type(
        self,
        classification: Dict[str, Any],
        content_analysis: Dict[str, Any],
        context_factors: Dict[str, Any]
    ) -> MessageType:
        """Determine the overall message type."""
        
        purpose = classification['purpose']
        toxicity = classification['toxicity']
        
        if toxicity in ['high', 'severe']:
            return MessageType.TOXIC
        elif purpose == 'spam' or content_analysis.get('likely_spam'):
            return MessageType.SPAM
        elif purpose == 'question' or content_analysis.get('has_questions'):
            return MessageType.QUESTION
        elif purpose == 'complaint':
            return MessageType.COMPLAINT
        elif context_factors.get('support_channel') or classification['intent'] == 'help':
            return MessageType.SUPPORT_REQUEST
        elif purpose in ['greeting', 'social'] or context_factors.get('social_channel'):
            return MessageType.SOCIAL
        elif context_factors.get('announcement_channel'):
            return MessageType.ANNOUNCEMENT
        else:
            return MessageType.REGULAR
    
    def _create_fallback_classification(self, signal: Signal) -> EnhancedClassificationResult:
        """Create a fallback classification when all else fails."""
        return EnhancedClassificationResult(
            purpose='other',
            sentiment='neutral',
            urgency='low',
            toxicity='none',
            intent='other',
            confidence=0.1,
            reasoning='Fallback classification due to processing error',
            message_type=MessageType.REGULAR,
            content_analysis={'length': len(signal.content)},
            context_factors={}
        )
    
    def _update_stats(self, result: EnhancedClassificationResult):
        """Update classification statistics."""
        # Update type counters
        if result.message_type:
            type_key = result.message_type.value
            if type_key not in self.stats['classification_by_type']:
                self.stats['classification_by_type'][type_key] = 0
            self.stats['classification_by_type'][type_key] += 1
        
        # Update confidence distribution
        if result.confidence >= 0.8:
            self.stats['confidence_distribution']['high'] += 1
        elif result.confidence >= 0.5:
            self.stats['confidence_distribution']['medium'] += 1
        else:
            self.stats['confidence_distribution']['low'] += 1
        
        # Update average processing time
        if result.processing_time:
            current_avg = self.stats['average_processing_time']
            total = self.stats['total_classifications']
            self.stats['average_processing_time'] = (
                (current_avg * (total - 1) + result.processing_time) / total
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification service statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset classification statistics."""
        for key in self.stats:
            if isinstance(self.stats[key], dict):
                self.stats[key] = {}
            else:
                self.stats[key] = 0
        
        logger.info("📊 Classification stats reset")


# Global service instance
_classification_service: Optional[MessageClassificationService] = None


async def get_classification_service() -> MessageClassificationService:
    """Get the global classification service instance."""
    global _classification_service
    if _classification_service is None:
        from app.llm.client import get_llm_client
        llm_client = await get_llm_client()
        _classification_service = MessageClassificationService(llm_client)
    return _classification_service


async def init_classification_service(
    llm_client: Optional[OpenAILLMClient] = None
) -> MessageClassificationService:
    """Initialize and return the classification service."""
    global _classification_service
    _classification_service = MessageClassificationService(llm_client)
    return _classification_service


async def classify_message(signal: Signal) -> EnhancedClassificationResult:
    """Convenience function to classify a message."""
    service = await get_classification_service()
    return await service.classify_message(signal)