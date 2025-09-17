"""
User Profile Aggregator for Discord Message Orchestration

This module aggregates classified messages to build user profiles over time,
determining when user behavior warrants invoking the expensive ReAct agent.
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque

from app.signals import Signal as BaseSignal, DiscordMessage
from app.llm.classification import UrgencyLevel, ToxicityLevel, PurposeCategory
from app.infra.db import get_database


logger = logging.getLogger(__name__)


class ActionTrigger(Enum):
    """Triggers that warrant invoking the ReAct agent."""
    QUESTION = "question"           # Direct questions needing answers
    COMPLAINT = "complaint"         # Complaints needing responses
    TOXIC_ESCALATION = "toxic_escalation"    # Escalating toxic behavior
    HIGH_URGENCY = "high_urgency"   # High urgency messages
    REPEAT_ISSUE = "repeat_issue"   # Repeated issues from same user
    NEW_USER_QUESTION = "new_user_question"  # Questions from new users


@dataclass
class UserProfile:
    """User behavior profile built from message history."""
    user_id: str
    username: str
    guild_id: str
    
    # Message statistics
    total_messages: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    # Classification aggregates
    purpose_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    urgency_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    toxicity_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sentiment_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Recent activity (last 24 hours)
    recent_messages: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_urgency_score: float = 0.0
    recent_toxicity_score: float = 0.0
    
    # Response tracking
    last_response_sent: Optional[datetime] = None
    response_cooldown_until: Optional[datetime] = None
    unanswered_questions: int = 0
    unresolved_complaints: int = 0
    
    # Behavioral flags
    is_new_user: bool = True
    is_frequent_questioner: bool = False
    is_problematic_user: bool = False
    requires_monitoring: bool = False


@dataclass
class AggregatedSignal:
    """Aggregated signal that may trigger ReAct agent."""
    user_id: str
    trigger: ActionTrigger
    urgency_score: float
    context: Dict[str, Any]
    original_signal: BaseSignal
    profile: UserProfile


class UserProfileAggregator:
    """
    Aggregates user behavior from classified messages to determine
    when the ReAct agent should be invoked for responses.
    """
    
    def __init__(self, 
                 profile_retention_days: int = 30,
                 response_cooldown_minutes: int = 30,
                 max_profiles_in_memory: int = 1000):
        """
        Initialize the aggregator.
        
        Args:
            profile_retention_days: How long to keep user profiles
            response_cooldown_minutes: Minimum time between responses to same user
            max_profiles_in_memory: Maximum user profiles to keep in memory
        """
        self.profile_retention_days = profile_retention_days
        self.response_cooldown_minutes = response_cooldown_minutes
        self.max_profiles_in_memory = max_profiles_in_memory
        
        # In-memory profile cache
        self._profiles: Dict[str, UserProfile] = {}
        self._profile_access_times: Dict[str, datetime] = {}
        
        # Statistics
        self._stats = {
            'messages_processed': 0,
            'profiles_created': 0,
            'profiles_updated': 0,
            'triggers_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_cleanup': datetime.now(timezone.utc)
        }
        
        logger.info(f"UserProfileAggregator initialized with {max_profiles_in_memory} max profiles")
    
    async def process_classified_message(self, signal: BaseSignal) -> Optional[AggregatedSignal]:
        """
        Process a classified message and update user profile.
        Returns AggregatedSignal if ReAct agent should be triggered.
        
        Args:
            signal: Classified message signal
            
        Returns:
            AggregatedSignal if action is warranted, None otherwise
        """
        try:
            self._stats['messages_processed'] += 1
            
            # Extract message details
            if not isinstance(signal, DiscordMessage):
                return None
                
            user_id = signal.author.get('user_id')
            if not user_id:
                logger.warning(f"No user_id found in signal {signal.signal_id} author data")
                return None
                
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            if isinstance(classification, str):
                try:
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    classification = {}
            
            # Get or create user profile
            profile = await self._get_or_create_profile(signal)
            
            # Update profile with new message
            await self._update_profile(profile, signal, classification)
            
            # Check if this message triggers ReAct agent
            trigger = await self._evaluate_trigger_conditions(profile, signal, classification)
            
            if trigger:
                self._stats['triggers_generated'] += 1
                
                return AggregatedSignal(
                    user_id=user_id,
                    trigger=trigger,
                    urgency_score=self._calculate_urgency_score(profile, classification),
                    context={
                        'profile_summary': self._get_profile_summary(profile),
                        'recent_activity': self._get_recent_activity_summary(profile),
                        'classification': classification
                    },
                    original_signal=signal,
                    profile=profile
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing classified message {signal.signal_id}: {e}")
            return None
    
    async def record_response_sent(self, user_id: str, response_signal: BaseSignal) -> None:
        """
        Record that a response was sent to a user, updating their profile
        to reduce urgency and set cooldown period.
        
        Args:
            user_id: User who received the response
            response_signal: The response signal sent
        """
        try:
            profile = self._profiles.get(user_id)
            if not profile:
                return
                
            now = datetime.now(timezone.utc)
            profile.last_response_sent = now
            profile.response_cooldown_until = now + timedelta(minutes=self.response_cooldown_minutes)
            
            # Reduce urgency scores after response
            profile.recent_urgency_score *= 0.5  # Decay urgency by half
            profile.unanswered_questions = max(0, profile.unanswered_questions - 1)
            profile.unresolved_complaints = max(0, profile.unresolved_complaints - 1)
            
            # Update profile in database
            await self._save_profile_to_db(profile)
            
            logger.debug(f"Recorded response sent to user {user_id}, set cooldown until {profile.response_cooldown_until}")
            
        except Exception as e:
            logger.error(f"Error recording response sent to {user_id}: {e}")
    
    async def _get_or_create_profile(self, signal: DiscordMessage) -> UserProfile:
        """Get existing user profile or create new one."""
        user_id = signal.author.get('user_id')
        if not user_id:
            raise ValueError(f"No user_id found in signal {signal.signal_id} author data")
        
        # Check memory cache first
        if user_id in self._profiles:
            self._stats['cache_hits'] += 1
            self._profile_access_times[user_id] = datetime.now(timezone.utc)
            return self._profiles[user_id]
        
        self._stats['cache_misses'] += 1
        
        # Try to load from database
        profile = await self._load_profile_from_db(user_id)
        
        if not profile:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                username=signal.author.get('username', 'Unknown'),
                guild_id=str(signal.context.get('guild_id', 'unknown')),
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc)
            )
            self._stats['profiles_created'] += 1
        else:
            self._stats['profiles_updated'] += 1
        
        # Add to memory cache
        self._profiles[user_id] = profile
        self._profile_access_times[user_id] = datetime.now(timezone.utc)
        
        # Clean up cache if needed
        await self._cleanup_profile_cache()
        
        return profile
    
    async def _update_profile(self, profile: UserProfile, signal: DiscordMessage, classification: Dict[str, Any]) -> None:
        """Update user profile with new message data."""
        now = datetime.now(timezone.utc)
        
        # Update basic stats
        profile.total_messages += 1
        profile.last_seen = now
        
        # Update classification counts
        if 'purpose' in classification:
            profile.purpose_counts[classification['purpose']] += 1
        if 'urgency' in classification:
            profile.urgency_counts[classification['urgency']] += 1
        if 'toxicity' in classification:
            profile.toxicity_counts[classification['toxicity']] += 1
        if 'sentiment' in classification:
            profile.sentiment_counts[classification['sentiment']] += 1
        
        # Add to recent messages
        message_summary = {
            'timestamp': now,
            'content': signal.content[:100],  # First 100 chars
            'classification': classification,
            'signal_id': signal.signal_id
        }
        profile.recent_messages.append(message_summary)
        
        # Update recent scores (weighted by recency)
        urgency_weight = self._get_urgency_weight(classification.get('urgency', 'low'))
        toxicity_weight = self._get_toxicity_weight(classification.get('toxicity', 'none'))
        
        # Decay existing scores and add new weighted scores
        profile.recent_urgency_score = (profile.recent_urgency_score * 0.9) + (urgency_weight * 0.1)
        profile.recent_toxicity_score = (profile.recent_toxicity_score * 0.9) + (toxicity_weight * 0.1)
        
        # Update behavioral counters
        purpose = classification.get('purpose', '').lower()
        if purpose == 'question':
            profile.unanswered_questions += 1
        elif purpose == 'complaint':
            profile.unresolved_complaints += 1
        
        # Update behavioral flags
        profile.is_new_user = profile.total_messages <= 5
        profile.is_frequent_questioner = profile.purpose_counts.get('question', 0) > 10
        profile.is_problematic_user = (
            profile.toxicity_counts.get('high', 0) > 3 or
            profile.recent_toxicity_score > 0.7
        )
        profile.requires_monitoring = (
            profile.is_problematic_user or 
            profile.unresolved_complaints > 2 or
            profile.recent_urgency_score > 0.8
        )
        
        # Save to database periodically
        if profile.total_messages % 10 == 0:  # Save every 10 messages
            await self._save_profile_to_db(profile)
    
    async def _evaluate_trigger_conditions(self, profile: UserProfile, signal: DiscordMessage, classification: Dict[str, Any]) -> Optional[ActionTrigger]:
        """
        Evaluate if this message should trigger the ReAct agent.
        
        Returns:
            ActionTrigger if agent should be invoked, None otherwise
        """
        # Check cooldown period
        now = datetime.now(timezone.utc)
        if profile.response_cooldown_until and now < profile.response_cooldown_until:
            return None  # Still in cooldown
        
        purpose = classification.get('purpose', '').lower()
        urgency = classification.get('urgency', 'low').lower()
        toxicity = classification.get('toxicity', 'none').lower()
        
        # High-priority immediate triggers
        if urgency == 'critical':
            return ActionTrigger.HIGH_URGENCY
            
        if purpose == 'complaint':
            return ActionTrigger.COMPLAINT
            
        # Questions from new users get priority
        if purpose == 'question' and profile.is_new_user:
            return ActionTrigger.NEW_USER_QUESTION
            
        # Regular questions if not answered recently
        if purpose == 'question' and profile.unanswered_questions >= 1:
            return ActionTrigger.QUESTION
            
        # Toxic behavior escalation
        if toxicity in ['high', 'severe'] or profile.recent_toxicity_score > 0.6:
            return ActionTrigger.TOXIC_ESCALATION
            
        # High urgency accumulated over recent messages
        if profile.recent_urgency_score > 0.7 and urgency in ['high', 'medium']:
            return ActionTrigger.HIGH_URGENCY
            
        # Repeat issues from problematic users
        if profile.requires_monitoring and urgency in ['high', 'medium']:
            return ActionTrigger.REPEAT_ISSUE
        
        return None  # No trigger
    
    def _calculate_urgency_score(self, profile: UserProfile, classification: Dict[str, Any]) -> float:
        """Calculate overall urgency score for prioritizing ReAct agent tasks."""
        base_urgency = self._get_urgency_weight(classification.get('urgency', 'low'))
        toxicity_boost = self._get_toxicity_weight(classification.get('toxicity', 'none')) * 0.3
        profile_boost = profile.recent_urgency_score * 0.2
        new_user_boost = 0.2 if profile.is_new_user else 0.0
        
        return min(1.0, base_urgency + toxicity_boost + profile_boost + new_user_boost)
    
    def _get_urgency_weight(self, urgency: str) -> float:
        """Convert urgency level to numeric weight."""
        weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        return weights.get(urgency.lower(), 0.2)
    
    def _get_toxicity_weight(self, toxicity: str) -> float:
        """Convert toxicity level to numeric weight."""
        weights = {
            'severe': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'mild': 0.3,
            'none': 0.0
        }
        return weights.get(toxicity.lower(), 0.0)
    
    def _get_profile_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Get summary of user profile for ReAct agent context."""
        return {
            'total_messages': profile.total_messages,
            'days_active': (datetime.now(timezone.utc) - profile.first_seen).days if profile.first_seen else 0,
            'is_new_user': profile.is_new_user,
            'is_frequent_questioner': profile.is_frequent_questioner,
            'is_problematic_user': profile.is_problematic_user,
            'unanswered_questions': profile.unanswered_questions,
            'unresolved_complaints': profile.unresolved_complaints,
            'recent_urgency_score': round(profile.recent_urgency_score, 2),
            'recent_toxicity_score': round(profile.recent_toxicity_score, 2),
            'primary_purpose': max(profile.purpose_counts, key=profile.purpose_counts.get) if profile.purpose_counts else 'unknown'
        }
    
    def _get_recent_activity_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Get summary of recent user activity."""
        recent_count = len(profile.recent_messages)
        if recent_count == 0:
            return {'message_count': 0}
            
        # Analyze recent message types
        recent_purposes = defaultdict(int)
        recent_urgencies = defaultdict(int)
        
        for msg in profile.recent_messages:
            classification = msg.get('classification', {})
            if 'purpose' in classification:
                recent_purposes[classification['purpose']] += 1
            if 'urgency' in classification:
                recent_urgencies[classification['urgency']] += 1
        
        return {
            'message_count': recent_count,
            'timespan_hours': 24,  # Recent messages are from last 24 hours
            'dominant_purpose': max(recent_purposes, key=recent_purposes.get) if recent_purposes else 'unknown',
            'urgent_message_count': recent_urgencies.get('high', 0) + recent_urgencies.get('critical', 0),
            'last_message_time': profile.recent_messages[-1]['timestamp'].isoformat() if profile.recent_messages else None
        }
    
    async def _load_profile_from_db(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from database."""
        try:
            # TODO: Implement database loading
            # For now, return None to create new profiles
            return None
        except Exception as e:
            logger.error(f"Error loading profile for {user_id} from database: {e}")
            return None
    
    async def _save_profile_to_db(self, profile: UserProfile) -> None:
        """Save user profile to database."""
        try:
            # TODO: Implement database saving
            # For now, just log that we would save
            logger.debug(f"Would save profile for user {profile.user_id} to database")
        except Exception as e:
            logger.error(f"Error saving profile for {profile.user_id} to database: {e}")
    
    async def _cleanup_profile_cache(self) -> None:
        """Clean up old profiles from memory cache."""
        if len(self._profiles) <= self.max_profiles_in_memory:
            return
            
        # Remove oldest accessed profiles
        sorted_profiles = sorted(
            self._profile_access_times.items(),
            key=lambda x: x[1]
        )
        
        profiles_to_remove = len(self._profiles) - self.max_profiles_in_memory
        for user_id, _ in sorted_profiles[:profiles_to_remove]:
            if user_id in self._profiles:
                # Save to database before removing
                await self._save_profile_to_db(self._profiles[user_id])
                del self._profiles[user_id]
                del self._profile_access_times[user_id]
        
        logger.debug(f"Cleaned up {profiles_to_remove} profiles from memory cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            **self._stats,
            'profiles_in_memory': len(self._profiles),
            'memory_usage_percent': (len(self._profiles) / self.max_profiles_in_memory) * 100
        }
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile for external inspection."""
        return self._profiles.get(user_id)


# Global aggregator instance
_aggregator: Optional[UserProfileAggregator] = None


def get_aggregator() -> UserProfileAggregator:
    """Get the global aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = UserProfileAggregator()
    return _aggregator