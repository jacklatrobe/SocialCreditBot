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
    
    # Social Credit Score (the overall "goodness" score)
    social_credit_score: float = 100.0  # Start at neutral 100, range 0-200
    
    # Response Urgency Score (how urgently they need a response)
    response_urgency_score: float = 0.0  # Range 0.0-1.0, higher = more urgent
    
    # Classification aggregates
    purpose_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    urgency_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    toxicity_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sentiment_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Recent activity (last 24 hours)
    recent_messages: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_urgency_score: float = 0.0  # Legacy field, kept for compatibility
    recent_toxicity_score: float = 0.0  # Legacy field, kept for compatibility
    
    # Response tracking
    last_response_sent: Optional[datetime] = None
    response_cooldown_until: Optional[datetime] = None
    unanswered_questions: int = 0
    unresolved_complaints: int = 0
    
    # Social Credit tracking
    consecutive_good_messages: int = 0  # Non-toxic, non-spam messages
    consecutive_bad_messages: int = 0   # Toxic or spam messages
    times_helped: int = 0  # Number of times bot responded helpfully
    
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
                 response_cooldown_minutes: int = 3,
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
        start_time = datetime.now(timezone.utc)
        try:
            self._stats['messages_processed'] += 1
            
            # Extract message details first for better logging
            if not isinstance(signal, DiscordMessage):
                logger.warning(f"⚠️ Signal {signal.signal_id} is not a DiscordMessage, skipping aggregation")
                return None
                
            user_id = signal.author.get('user_id')
            username = signal.author.get('username', 'unknown')
            if not user_id:
                logger.warning(f"⚠️ No user_id found in signal {signal.signal_id} from {username}")
                return None
                
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            if isinstance(classification, str):
                try:
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    classification = {}
            
            # Log detailed entry info
            purpose = classification.get('purpose', 'unknown')
            urgency = classification.get('urgency', 'unknown')
            toxicity = classification.get('toxicity', 'unknown')
            content_preview = signal.content[:50] + "..." if len(signal.content) > 50 else signal.content
            
            logger.info(f"🔍 Processing message from {username} ({user_id}): "
                       f"purpose={purpose}, urgency={urgency}, toxicity={toxicity}, "
                       f"content='{content_preview}'")
            
            # Get or create user profile
            profile_start = datetime.now(timezone.utc)
            profile = await self._get_or_create_profile(signal)
            profile_load_ms = (datetime.now(timezone.utc) - profile_start).total_seconds() * 1000
            logger.debug(f"📂 Profile loaded for {username} in {profile_load_ms:.1f}ms (total_messages={profile.total_messages})")
            
            # Update profile with new message
            update_start = datetime.now(timezone.utc)
            await self._update_profile(profile, signal, classification)
            update_ms = (datetime.now(timezone.utc) - update_start).total_seconds() * 1000
            
            logger.debug(f"📈 Profile updated for {username} in {update_ms:.1f}ms: "
                        f"social_credit={profile.social_credit_score:.1f}, "
                        f"response_urgency={profile.response_urgency_score:.3f}, "
                        f"consecutive_good={profile.consecutive_good_messages}, "
                        f"consecutive_bad={profile.consecutive_bad_messages}")
            
            # Check if this message triggers ReAct agent
            trigger_start = datetime.now(timezone.utc)
            trigger = await self._evaluate_trigger_conditions(profile, signal, classification)
            trigger_ms = (datetime.now(timezone.utc) - trigger_start).total_seconds() * 1000
            
            total_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            if trigger:
                self._stats['triggers_generated'] += 1
                
                logger.info(f"🚨 AGENT TRIGGERED for {username}: {trigger.value} "
                           f"(urgency={profile.response_urgency_score:.3f}, total_time={total_ms:.1f}ms)")
                
                # Build conversation context from recent messages
                conversation_context = self._build_conversation_context(profile)
                
                return AggregatedSignal(
                    user_id=user_id,
                    trigger=trigger,
                    urgency_score=self._calculate_urgency_score(profile, classification),
                    context={
                        'profile_summary': self._get_profile_summary(profile),
                        'recent_activity': self._get_recent_activity_summary(profile),
                        'conversation_history': conversation_context,  # Add conversation history
                        'classification': self._clean_classification(classification)  # Clean legacy fields
                    },
                    original_signal=signal,
                    profile=profile
                )
            else:
                # Log why agent was NOT triggered - this is key info you wanted
                logger.info(f"⏳ NO TRIGGER for {username}: "
                           f"urgency={profile.response_urgency_score:.3f}, "
                           f"social_credit={profile.social_credit_score:.1f}, "
                           f"cooldown_until={profile.response_cooldown_until.strftime('%H:%M:%S') if profile.response_cooldown_until else 'none'}, "
                           f"unanswered_q={profile.unanswered_questions}, "
                           f"unresolved_complaints={profile.unresolved_complaints}, "
                           f"total_time={total_ms:.1f}ms")
            
            return None
            
        except Exception as e:
            total_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000 if 'start_time' in locals() else 0
            username = signal.author.get('username', 'unknown') if hasattr(signal, 'author') else 'unknown'
            logger.error(f"❌ Error processing message from {username} after {total_ms:.1f}ms: {e}", exc_info=True)
            return None
    
    async def record_response_sent(self, user_id: str, response_signal: BaseSignal) -> None:
        """
        Record that a response was sent to a user, updating their profile
        to reduce urgency and boost social credit for receiving help.
        
        Args:
            user_id: User who received the response
            response_signal: The response signal sent
        """
        try:
            profile = self._profiles.get(user_id)
            if not profile:
                logger.warning(f"⚠️ No profile found for user {user_id} when recording response sent")
                return
            
            logger.info(f"✅ Recording response sent to {profile.username}: "
                       f"previous_urgency={profile.response_urgency_score:.3f}, "
                       f"previous_social_credit={profile.social_credit_score:.1f}")
                
            now = datetime.now(timezone.utc)
            profile.last_response_sent = now
            profile.response_cooldown_until = now + timedelta(minutes=self.response_cooldown_minutes)
            
            # Reduce response urgency after helping the user
            old_urgency = profile.response_urgency_score
            profile.response_urgency_score = profile.response_urgency_score * 0.75
            
            # Boost social credit for receiving help (encourages positive behavior)
            old_credit = profile.social_credit_score
            profile.social_credit_score = min(200, profile.social_credit_score + 2)
            
            # Reset consecutive message counters
            profile.consecutive_good_messages = 0
            profile.consecutive_bad_messages = 0
            profile.unanswered_questions = 0
            profile.unresolved_complaints = 0
            
            # Update profile in database
            await self._save_profile_to_db(profile)
            
            logger.info(f"🎉 Response recorded for {profile.username}: "
                       f"urgency={old_urgency:.3f}→{profile.response_urgency_score:.3f}, "
                       f"social_credit={old_credit:.1f}→{profile.social_credit_score:.1f}, "
                       f"cooldown_until={profile.response_cooldown_until.strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error recording response sent to {user_id}: {e}")
    
    async def _get_or_create_profile(self, signal: DiscordMessage) -> UserProfile:
        """Get existing user profile or create new one."""
        user_id = signal.author.get('user_id')
        username = signal.author.get('username', 'Unknown')
        if not user_id:
            raise ValueError(f"No user_id found in signal {signal.signal_id} author data")
        
        # Check memory cache first
        if user_id in self._profiles:
            self._stats['cache_hits'] += 1
            self._profile_access_times[user_id] = datetime.now(timezone.utc)
            logger.debug(f"🎯 Cache HIT for {username} ({user_id}) - profile in memory")
            return self._profiles[user_id]
        
        self._stats['cache_misses'] += 1
        logger.debug(f"💾 Cache MISS for {username} ({user_id}) - loading from DB")
        
        # Try to load from database
        db_start = datetime.now(timezone.utc)
        profile = await self._load_profile_from_db(user_id)
        db_load_ms = (datetime.now(timezone.utc) - db_start).total_seconds() * 1000
        
        if not profile:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                username=username,
                guild_id=str(signal.context.get('guild_id', 'unknown')),
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc)
            )
            self._stats['profiles_created'] += 1
            logger.info(f"👤 NEW PROFILE created for {username} ({user_id}) in {db_load_ms:.1f}ms")
        else:
            self._stats['profiles_updated'] += 1
            days_since_last = (datetime.now(timezone.utc) - profile.last_seen).days if profile.last_seen else 0
            logger.debug(f"📂 Profile LOADED for {username} from DB in {db_load_ms:.1f}ms "
                        f"(total_messages={profile.total_messages}, last_seen={days_since_last} days ago)")
        
        # Add to memory cache
        self._profiles[user_id] = profile
        self._profile_access_times[user_id] = datetime.now(timezone.utc)
        
        # Clean up cache if needed
        await self._cleanup_profile_cache()
        
        return profile
    
    async def _update_profile(self, profile: UserProfile, signal: DiscordMessage, classification: Dict[str, Any]) -> None:
        """Update user profile with new message data and social credit scoring."""
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
        
        # Update legacy scores (for backward compatibility)
        urgency_weight = self._get_urgency_weight(classification.get('urgency', 'low'))
        toxicity_weight = self._get_toxicity_weight(classification.get('toxicity', 'none'))
        profile.recent_urgency_score = (profile.recent_urgency_score * 0.9) + (urgency_weight * 0.1)
        profile.recent_toxicity_score = (profile.recent_toxicity_score * 0.9) + (toxicity_weight * 0.1)
        
        # === SOCIAL CREDIT SYSTEM UPDATE ===
        await self._update_social_credit_score(profile, classification)
        
        # === RESPONSE URGENCY SYSTEM UPDATE ===
        await self._update_response_urgency_score(profile, classification)
        
        # Update behavioral flags
        profile.is_new_user = profile.total_messages <= 5
        profile.is_frequent_questioner = profile.purpose_counts.get('question', 0) > 10
        profile.is_problematic_user = (
            profile.social_credit_score < 70 or  # Low social credit
            profile.toxicity_counts.get('high', 0) > 3 or
            profile.recent_toxicity_score > 0.7
        )
        profile.requires_monitoring = (
            profile.is_problematic_user or 
            profile.unresolved_complaints > 2 or
            profile.response_urgency_score > 0.8  # High response urgency
        )
        
        # Save to database periodically
        if profile.total_messages % 10 == 0:  # Save every 10 messages
            await self._save_profile_to_db(profile)
    
    async def _update_social_credit_score(self, profile: UserProfile, classification: Dict[str, Any]) -> None:
        """
        Update user's social credit score based on their message behavior.
        
        Social Credit Rules:
        - Good messages (questions, non-toxic chat): +1 to +3 points
        - Bad messages (toxic, spam): -5 to -15 points
        - Range: 0-200, start at 100
        """
        purpose = classification.get('purpose', '').lower()
        toxicity = classification.get('toxicity', 'none').lower()
        message_type = classification.get('message_type', '').lower()
        
        credit_change = 0
        
        # Negative behaviors (decrease social credit)
        if toxicity in ['severe', 'high']:
            credit_change = -15
            profile.consecutive_bad_messages += 1
            profile.consecutive_good_messages = 0
        elif toxicity in ['moderate', 'medium']:
            credit_change = -8
            profile.consecutive_bad_messages += 1
            profile.consecutive_good_messages = 0
        elif message_type == 'spam':
            credit_change = -10
            profile.consecutive_bad_messages += 1
            profile.consecutive_good_messages = 0
        else:
            # Positive behaviors (increase social credit)
            if purpose == 'question':
                credit_change = 3  # Questions are valuable
            elif purpose in ['greeting', 'social'] and toxicity == 'none':
                credit_change = 1  # Friendly chat
            elif toxicity == 'none':
                credit_change = 2  # Any non-toxic message
            
            if credit_change > 0:
                profile.consecutive_good_messages += 1
                profile.consecutive_bad_messages = 0
        
        # Apply consecutive message bonuses/penalties
        if profile.consecutive_good_messages > 5:
            credit_change += 1  # Bonus for sustained good behavior
        elif profile.consecutive_bad_messages > 3:
            credit_change -= 2  # Additional penalty for sustained bad behavior
        
        # Update score with bounds checking
        old_score = profile.social_credit_score
        profile.social_credit_score = max(0.0, min(200.0, profile.social_credit_score + credit_change))
        
        if credit_change != 0:
            logger.debug(f"💳 Social credit for {profile.username}: {old_score:.1f} {credit_change:+d} → {profile.social_credit_score:.1f} "
                        f"(good_streak={profile.consecutive_good_messages}, bad_streak={profile.consecutive_bad_messages})")
        else:
            logger.debug(f"💳 Social credit unchanged for {profile.username}: {profile.social_credit_score:.1f}")
    
    async def _update_response_urgency_score(self, profile: UserProfile, classification: Dict[str, Any]) -> None:
        """
        Update user's response urgency score based on their immediate needs.
        
        Response Urgency Rules:
        - Questions: Rapid increase (1-2 messages to trigger)
        - Toxicity: Gradual increase (watch and wait)
        - Good behavior: Decrease urgency
        - Getting responses: Reset urgency
        """
        purpose = classification.get('purpose', '').lower()
        toxicity = classification.get('toxicity', 'none').lower()
        urgency = classification.get('urgency', 'low').lower()
        
        urgency_change = 0.0
        
        # Question handling - fast response
        if purpose == 'question':
            if profile.is_new_user:
                urgency_change += 0.9  # New users get priority
            else:
                urgency_change += 0.7   # Regular users still get quick responses
            profile.unanswered_questions += 1
        
        # Complaint handling - moderate response
        elif purpose == 'complaint':
            urgency_change += 0.4
            profile.unresolved_complaints += 1
        
        # Toxicity handling - gradual increase (watch and wait)
        elif toxicity in ['high', 'severe']:
            urgency_change = 0.4  # Slower build-up for toxic behavior
        elif toxicity in ['moderate', 'medium']:
            urgency_change = 0.1
        
        # High urgency classification
        elif urgency in ['high', 'critical']:
            urgency_change = 0.8
        
        # Good behavior slightly reduces urgency
        elif toxicity == 'none' and purpose in ['greeting', 'social']:
            urgency_change = -0.1  # Small decrease for friendly behavior
        
        # Apply urgency change with decay
        old_urgency = profile.response_urgency_score
        profile.response_urgency_score = max(0.0, min(1.0, 
            profile.response_urgency_score * 0.95 + urgency_change  # Slight natural decay + new urgency
        ))
        
        if urgency_change != 0.0:
            logger.debug(f"📈 Response urgency for {profile.username}: {old_urgency:.3f} {urgency_change:+.2f} → {profile.response_urgency_score:.3f} "
                        f"(unanswered_q={profile.unanswered_questions}, complaints={profile.unresolved_complaints})")
        else:
            logger.debug(f"📈 Response urgency decayed for {profile.username}: {old_urgency:.3f} → {profile.response_urgency_score:.3f}")
    
    async def _evaluate_trigger_conditions(self, profile: UserProfile, signal: DiscordMessage, classification: Dict[str, Any]) -> Optional[ActionTrigger]:
        """
        Evaluate if this message should trigger the ReAct agent.
        
        Uses the new response urgency score system:
        - Questions trigger quickly (1-2 messages)
        - Toxicity builds slowly (watch and wait)
        - Social credit influences thresholds
        
        Returns:
            ActionTrigger if agent should be invoked, None otherwise
        """
        username = signal.author.get('username', 'unknown')
        
        # Check cooldown period
        now = datetime.now(timezone.utc)
        if profile.response_cooldown_until and now < profile.response_cooldown_until:
            remaining_cooldown = (profile.response_cooldown_until - now).total_seconds()
            logger.debug(f"⏰ {username} still in cooldown for {remaining_cooldown:.0f}s")
            return None  # Still in cooldown
        
        purpose = classification.get('purpose', '').lower()
        urgency = classification.get('urgency', 'low').lower()
        toxicity = classification.get('toxicity', 'none').lower()
        
        logger.debug(f"🎯 Evaluating triggers for {username}: purpose={purpose}, urgency={urgency}, toxicity={toxicity}, "
                    f"response_urgency={profile.response_urgency_score:.3f}, social_credit={profile.social_credit_score:.1f}")
        
        # === PRIMARY TRIGGER LOGIC BASED ON RESPONSE URGENCY ===
        
        # Questions: Fast response threshold
        if purpose == 'question':
            question_threshold = 0.6 if profile.is_new_user else 0.7
            logger.debug(f"🔍 Question check: urgency={profile.response_urgency_score:.3f} vs threshold={question_threshold} (new_user={profile.is_new_user})")
            if profile.response_urgency_score >= question_threshold:
                trigger = ActionTrigger.NEW_USER_QUESTION if profile.is_new_user else ActionTrigger.QUESTION
                logger.info(f"✅ TRIGGER: {trigger.value} for {username} (question threshold exceeded)")
                return trigger
        
        # Complaints: Medium response threshold  
        elif purpose == 'complaint':
            logger.debug(f"😠 Complaint check: urgency={profile.response_urgency_score:.3f} vs threshold=0.5")
            if profile.response_urgency_score >= 0.5:
                logger.info(f"✅ TRIGGER: {ActionTrigger.COMPLAINT.value} for {username} (complaint threshold exceeded)")
                return ActionTrigger.COMPLAINT
        
        # Toxic behavior: Higher threshold (watch and wait)
        elif toxicity in ['high', 'severe']:
            toxic_threshold = 0.8
            # Lower threshold for users with poor social credit
            if profile.social_credit_score < 50:
                toxic_threshold = 0.6
            logger.debug(f"☠️ Toxic check: urgency={profile.response_urgency_score:.3f} vs threshold={toxic_threshold} (social_credit={profile.social_credit_score:.1f})")
            if profile.response_urgency_score >= toxic_threshold:
                logger.info(f"✅ TRIGGER: {ActionTrigger.TOXIC_ESCALATION.value} for {username} (toxic threshold exceeded)")
                return ActionTrigger.TOXIC_ESCALATION
        
        # Critical urgency always triggers immediately
        if urgency == 'critical':
            logger.info(f"✅ TRIGGER: {ActionTrigger.HIGH_URGENCY.value} for {username} (critical urgency)")
            return ActionTrigger.HIGH_URGENCY
        if urgency == 'critical':
            return ActionTrigger.HIGH_URGENCY
        
        # High overall urgency based on accumulated score
        if profile.response_urgency_score >= 0.9:
            logger.info(f"✅ TRIGGER: {ActionTrigger.HIGH_URGENCY.value} for {username} (high accumulated urgency)")
            return ActionTrigger.HIGH_URGENCY
        
        # Repeat issues for users who need monitoring
        if profile.requires_monitoring and profile.response_urgency_score >= 0.7:
            logger.info(f"✅ TRIGGER: {ActionTrigger.REPEAT_ISSUE.value} for {username} (monitoring user with high urgency)")
            return ActionTrigger.REPEAT_ISSUE
        
        # Log why no trigger occurred
        logger.debug(f"❌ No trigger conditions met for {username}: "
                    f"response_urgency={profile.response_urgency_score:.3f}, "
                    f"requires_monitoring={profile.requires_monitoring}, "
                    f"unanswered_q={profile.unanswered_questions}, "
                    f"unresolved_complaints={profile.unresolved_complaints}")
        
        return None  # No trigger
    
    def _calculate_urgency_score(self, profile: UserProfile, classification: Dict[str, Any]) -> float:
        """Calculate overall urgency score for prioritizing ReAct agent tasks (legacy method)."""
        # Return the new response urgency score - this replaces the complex calculation
        # and uses the improved scoring system implemented in _update_response_urgency_score
        return profile.response_urgency_score
    
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
    
    def _build_conversation_context(self, profile: UserProfile, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Build conversation history context from user's recent messages.
        
        Args:
            profile: User profile containing recent messages
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of message contexts with content, timestamp, and classification
        """
        conversation = []
        
        # Get the most recent messages, up to max_messages
        recent_messages = list(profile.recent_messages)[-max_messages:] if profile.recent_messages else []
        
        for msg_data in recent_messages:
            conversation.append({
                'timestamp': msg_data.get('timestamp', '').isoformat() if hasattr(msg_data.get('timestamp', ''), 'isoformat') else str(msg_data.get('timestamp', '')),
                'content': msg_data.get('content', ''),
                'classification': msg_data.get('classification', {}),
                'signal_id': msg_data.get('signal_id', '')
            })
        
        return conversation
    
    def _clean_classification(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove legacy classification fields that shouldn't influence the ReAct agent.
        
        Args:
            classification: Raw classification data
            
        Returns:
            Cleaned classification without legacy fields
        """
        cleaned = classification.copy()
        
        # Remove fields that the ReAct agent shouldn't see
        legacy_fields = ['requires_response', 'confidence', 'should_respond']
        
        for field in legacy_fields:
            cleaned.pop(field, None)
        
        return cleaned
    
    def _get_profile_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Get summary of user profile for ReAct agent context."""
        return {
            'total_messages': profile.total_messages,
            'days_active': (datetime.now(timezone.utc) - profile.first_seen).days if profile.first_seen else 0,
            'is_new_user': profile.is_new_user,
            'is_frequent_questioner': profile.is_frequent_questioner,
            'is_problematic_user': profile.is_problematic_user,
            'consecutive_good_messages': profile.consecutive_good_messages,
            'consecutive_bad_messages': profile.consecutive_bad_messages,
            'unanswered_questions': profile.unanswered_questions,
            'unresolved_complaints': profile.unresolved_complaints,
            
            # New social credit system scores
            'social_credit_score': profile.social_credit_score,
            'response_urgency_score': round(profile.response_urgency_score, 3),
            
            # Legacy scores for backward compatibility
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
            'questions_count': recent_purposes.get('question', 0),
            'complaints_count': recent_purposes.get('complaint', 0),
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