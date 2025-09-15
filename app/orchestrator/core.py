"""
Core Orchestrator for Discord Observer/Orchestrator

This module implements the central Orchestrator component that:
1. Subscribes to SIGNAL_CLASSIFIED events from the LLM Observer
2. Applies orchestration logic based on message classification
3. Determines appropriate responses and actions
4. Coordinates between different system components
5. Manages the overall message processing workflow

The Orchestrator acts as the brain of the system, making decisions about how
to respond to classified Discord messages based on business rules and
system configuration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timezone
from enum import Enum

from app.signals import Signal as BaseSignal, DiscordMessage
from app.infra.bus import signal_bus
from app.config import get_settings


logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the orchestrator can take."""
    NO_ACTION = "no_action"
    RESPOND = "respond"
    ESCALATE = "escalate"
    MODERATE = "moderate"
    LOG_ONLY = "log_only"
    NOTIFY_ADMIN = "notify_admin"


class ResponsePriority(Enum):
    """Priority levels for responses."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class OrchestrationRule:
    """
    A rule that defines how to respond to specific types of classified messages.
    
    Rules are evaluated in priority order to determine the appropriate action
    for a given message classification.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        conditions: Dict[str, Any],
        action_type: ActionType,
        response_template: Optional[str] = None,
        priority: ResponsePriority = ResponsePriority.NORMAL,
        enabled: bool = True
    ):
        """
        Initialize an orchestration rule.
        
        Args:
            name: Human-readable name for the rule
            description: Description of what the rule does
            conditions: Dictionary of conditions that must be met
            action_type: Type of action to take when conditions are met
            response_template: Template for generating responses (if applicable)
            priority: Priority level for this rule
            enabled: Whether this rule is currently active
        """
        self.name = name
        self.description = description
        self.conditions = conditions
        self.action_type = action_type
        self.response_template = response_template
        self.priority = priority
        self.enabled = enabled
        self.created_at = datetime.now(timezone.utc)
        self.usage_count = 0
        self.last_used = None
    
    def matches(self, classification: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Check if this rule matches the given classification and context.
        
        Args:
            classification: Classification results from LLM
            context: Message context information
            
        Returns:
            True if all conditions are met
        """
        if not self.enabled:
            return False
        
        try:
            for condition_key, condition_value in self.conditions.items():
                if not self._evaluate_condition(condition_key, condition_value, classification, context):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            return False
    
    def _evaluate_condition(
        self, 
        condition_key: str, 
        condition_value: Any, 
        classification: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single condition.
        
        Args:
            condition_key: The condition key (e.g., 'message_type', 'confidence')
            condition_value: Expected value or criteria
            classification: Classification results
            context: Message context
            
        Returns:
            True if condition is met
        """
        # Handle classification-based conditions
        if condition_key.startswith('classification.'):
            classification_key = condition_key[13:]  # Remove 'classification.' prefix
            actual_value = classification.get(classification_key)
            return self._compare_values(actual_value, condition_value)
        
        # Handle context-based conditions
        elif condition_key.startswith('context.'):
            context_key = condition_key[8:]  # Remove 'context.' prefix
            actual_value = context.get(context_key)
            return self._compare_values(actual_value, condition_value)
        
        # Handle direct conditions
        else:
            if condition_key in classification:
                actual_value = classification[condition_key]
            elif condition_key in context:
                actual_value = context[condition_key]
            else:
                return False
            
            return self._compare_values(actual_value, condition_value)
    
    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """Compare actual value against expected value or criteria."""
        if isinstance(expected, dict):
            # Handle complex conditions like {"min": 0.8} or {"in": ["value1", "value2"]}
            if "min" in expected:
                return isinstance(actual, (int, float)) and actual >= expected["min"]
            elif "max" in expected:
                return isinstance(actual, (int, float)) and actual <= expected["max"]
            elif "in" in expected:
                return actual in expected["in"]
            elif "not_in" in expected:
                return actual not in expected["not_in"]
            elif "equals" in expected:
                return actual == expected["equals"]
            elif "contains" in expected:
                return isinstance(actual, str) and expected["contains"] in actual
            else:
                return False
        else:
            # Simple equality check
            return actual == expected
    
    def execute(self) -> Dict[str, Any]:
        """
        Mark this rule as executed and return action details.
        
        Returns:
            Dictionary containing action details
        """
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
        
        return {
            "rule_name": self.name,
            "action_type": self.action_type.value,
            "response_template": self.response_template,
            "priority": self.priority.value,
            "executed_at": self.last_used.isoformat()
        }


class MessageOrchestrator:
    """
    Main orchestrator that coordinates message processing and response generation.
    
    The orchestrator receives classified messages and applies business rules to
    determine appropriate actions. It manages the workflow between classification
    and response generation.
    """
    
    def __init__(self):
        """Initialize the message orchestrator."""
        self.settings = get_settings()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._rules: List[OrchestrationRule] = []
        self._action_handlers: Dict[ActionType, Callable] = {}
        self._stats = {
            'messages_orchestrated': 0,
            'actions_taken': 0,
            'rules_executed': 0,
            'errors': 0,
            'start_time': None,
            'last_processed': None
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Initialize action handlers
        self._initialize_action_handlers()
        
        logger.info("Message Orchestrator initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default orchestration rules."""
        
        # Rule for high-confidence questions
        self.add_rule(OrchestrationRule(
            name="High Confidence Questions",
            description="Respond to questions with high confidence classification",
            conditions={
                "message_type": "question",
                "confidence": {"min": 0.8}
            },
            action_type=ActionType.RESPOND,
            response_template="helpful_response",
            priority=ResponsePriority.NORMAL
        ))
        
        # Rule for complaints
        self.add_rule(OrchestrationRule(
            name="Complaint Handling",
            description="Handle user complaints and escalate if necessary",
            conditions={
                "message_type": "complaint",
                "confidence": {"min": 0.7}
            },
            action_type=ActionType.ESCALATE,
            priority=ResponsePriority.HIGH
        ))
        
        # Rule for spam detection
        self.add_rule(OrchestrationRule(
            name="Spam Detection",
            description="Moderate spam messages",
            conditions={
                "message_type": "spam",
                "confidence": {"min": 0.6}
            },
            action_type=ActionType.MODERATE,
            priority=ResponsePriority.URGENT
        ))
        
        # Rule for toxic content
        self.add_rule(OrchestrationRule(
            name="Toxic Content",
            description="Handle toxic messages with moderation",
            conditions={
                "toxicity": {"min": 0.8}
            },
            action_type=ActionType.MODERATE,
            priority=ResponsePriority.URGENT
        ))
        
        # Rule for social messages
        self.add_rule(OrchestrationRule(
            name="Social Interactions",
            description="Respond to friendly social messages",
            conditions={
                "message_type": "social",
                "confidence": {"min": 0.7}
            },
            action_type=ActionType.RESPOND,
            response_template="social_response",
            priority=ResponsePriority.LOW
        ))
        
        # Default rule - log everything else
        self.add_rule(OrchestrationRule(
            name="Default Logging",
            description="Log all other messages for analysis",
            conditions={},  # Matches everything if no other rules match
            action_type=ActionType.LOG_ONLY,
            priority=ResponsePriority.LOW
        ))
        
        logger.info(f"Initialized {len(self._rules)} default orchestration rules")
    
    def _initialize_action_handlers(self) -> None:
        """Initialize handlers for different action types."""
        self._action_handlers = {
            ActionType.NO_ACTION: self._handle_no_action,
            ActionType.RESPOND: self._handle_respond,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.MODERATE: self._handle_moderate,
            ActionType.LOG_ONLY: self._handle_log_only,
            ActionType.NOTIFY_ADMIN: self._handle_notify_admin
        }
    
    def add_rule(self, rule: OrchestrationRule) -> None:
        """
        Add a new orchestration rule.
        
        Args:
            rule: The rule to add
        """
        self._rules.append(rule)
        # Sort rules by priority (urgent first)
        priority_order = {
            ResponsePriority.URGENT: 0,
            ResponsePriority.HIGH: 1,
            ResponsePriority.NORMAL: 2,
            ResponsePriority.LOW: 3
        }
        self._rules.sort(key=lambda r: priority_order.get(r.priority, 3))
        
        logger.debug(f"Added orchestration rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.debug(f"Removed orchestration rule: {rule_name}")
                return True
        
        logger.warning(f"Rule not found for removal: {rule_name}")
        return False
    
    def get_rules(self) -> List[OrchestrationRule]:
        """Get all orchestration rules."""
        return self._rules.copy()
    
    async def start(self) -> None:
        """
        Start the orchestrator to begin processing classified signals.
        """
        if self._running:
            logger.warning("Message Orchestrator is already running")
            return
        
        self._running = True
        self._stats['start_time'] = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._process_classified_signals())
        
        logger.info("Message Orchestrator started")
    
    async def stop(self) -> None:
        """
        Stop the orchestrator and clean up resources.
        """
        if not self._running:
            logger.warning("Message Orchestrator is not running")
            return
        
        self._running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._task = None
        logger.info("Message Orchestrator stopped")
    
    async def _process_classified_signals(self) -> None:
        """
        Main processing loop for classified signals.
        """
        logger.info("Orchestrator signal processing started")
        
        try:
            while self._running:
                try:
                    # Wait for classified signals with timeout
                    signal = await asyncio.wait_for(
                        signal_bus.subscribe("SIGNAL_CLASSIFIED"),
                        timeout=1.0
                    )
                    
                    if signal and isinstance(signal, BaseSignal):
                        await self._orchestrate_message(signal)
                        
                except asyncio.TimeoutError:
                    # Timeout is expected - allows checking if we should continue
                    continue
                except Exception as e:
                    logger.error(f"Error in orchestration loop: {e}")
                    self._stats['errors'] += 1
                    await asyncio.sleep(1)  # Brief pause on error
                    
        except asyncio.CancelledError:
            logger.info("Orchestrator signal processing cancelled")
        except Exception as e:
            logger.error(f"Fatal error in orchestrator: {e}")
        finally:
            logger.info("Orchestrator signal processing stopped")
    
    async def _orchestrate_message(self, signal: BaseSignal) -> None:
        """
        Orchestrate a single classified message.
        
        Args:
            signal: The classified signal to orchestrate
        """
        try:
            self._stats['messages_orchestrated'] += 1
            self._stats['last_processed'] = datetime.now(timezone.utc)
            
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            logger.debug(f"Orchestrating signal {signal.signal_id} with classification: {classification}")
            
            # Find matching rule
            matching_rule = self._find_matching_rule(classification, signal.context)
            
            if matching_rule:
                # Execute the rule
                action_details = matching_rule.execute()
                self._stats['rules_executed'] += 1
                
                logger.debug(f"Rule '{matching_rule.name}' matched for signal {signal.signal_id}")
                
                # Execute the action
                await self._execute_action(signal, matching_rule, action_details)
                self._stats['actions_taken'] += 1
                
            else:
                logger.warning(f"No matching rule found for signal {signal.signal_id}")
                # Use default log-only action
                await self._handle_log_only(signal, {})
                
        except Exception as e:
            logger.error(f"Error orchestrating signal {signal.signal_id}: {e}")
            self._stats['errors'] += 1
    
    def _find_matching_rule(self, classification: Dict[str, Any], context: Dict[str, Any]) -> Optional[OrchestrationRule]:
        """
        Find the first rule that matches the classification and context.
        
        Args:
            classification: Classification results
            context: Message context
            
        Returns:
            Matching rule or None if no match found
        """
        for rule in self._rules:
            if rule.matches(classification, context):
                return rule
        
        return None
    
    async def _execute_action(
        self, 
        signal: BaseSignal, 
        rule: OrchestrationRule, 
        action_details: Dict[str, Any]
    ) -> None:
        """
        Execute the action determined by the orchestration rule.
        
        Args:
            signal: The signal being processed
            rule: The rule that matched
            action_details: Details about the action to take
        """
        action_type = rule.action_type
        handler = self._action_handlers.get(action_type)
        
        if handler:
            await handler(signal, action_details)
        else:
            logger.error(f"No handler found for action type: {action_type}")
    
    # Action handlers
    async def _handle_no_action(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle no action - just log and continue."""
        logger.debug(f"No action taken for signal {signal.signal_id}")
    
    async def _handle_respond(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle response generation."""
        logger.info(f"Generating response for signal {signal.signal_id}")
        
        # Create response signal for the responder tool
        response_signal = self._create_response_signal(signal, action_details)
        await signal_bus.publish("SIGNAL_RESPOND", response_signal)
    
    async def _handle_escalate(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle escalation to human moderators."""
        logger.info(f"Escalating signal {signal.signal_id}")
        
        # Create escalation signal
        escalation_signal = self._create_escalation_signal(signal, action_details)
        await signal_bus.publish("SIGNAL_ESCALATE", escalation_signal)
    
    async def _handle_moderate(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle moderation actions."""
        logger.info(f"Moderating signal {signal.signal_id}")
        
        # Create moderation signal
        moderation_signal = self._create_moderation_signal(signal, action_details)
        await signal_bus.publish("SIGNAL_MODERATE", moderation_signal)
    
    async def _handle_log_only(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle logging without further action."""
        logger.info(f"Logging signal {signal.signal_id} for analysis")
        
        # Could save to database or analytics system here
        # For now, just ensure it's logged
    
    async def _handle_notify_admin(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle admin notification."""
        logger.info(f"Notifying admin about signal {signal.signal_id}")
        
        # Create admin notification signal
        notification_signal = self._create_notification_signal(signal, action_details)
        await signal_bus.publish("SIGNAL_NOTIFY_ADMIN", notification_signal)
    
    def _create_response_signal(self, original_signal: BaseSignal, action_details: Dict[str, Any]) -> BaseSignal:
        """Create a signal for response generation."""
        enhanced_context = original_signal.context.copy()
        enhanced_context.update({
            'orchestration': action_details,
            'action_type': 'respond',
            'orchestrated_at': datetime.now(timezone.utc).isoformat(),
            'processing_stage': 'orchestrated'
        })
        
        return BaseSignal(
            signal_id=f"resp_{original_signal.signal_id}",
            author=original_signal.author,
            content=original_signal.content,
            context=enhanced_context
        )
    
    def _create_escalation_signal(self, original_signal: BaseSignal, action_details: Dict[str, Any]) -> BaseSignal:
        """Create a signal for escalation."""
        enhanced_context = original_signal.context.copy()
        enhanced_context.update({
            'orchestration': action_details,
            'action_type': 'escalate',
            'escalated_at': datetime.now(timezone.utc).isoformat(),
            'processing_stage': 'escalated'
        })
        
        return BaseSignal(
            signal_id=f"esc_{original_signal.signal_id}",
            author=original_signal.author,
            content=original_signal.content,
            context=enhanced_context
        )
    
    def _create_moderation_signal(self, original_signal: BaseSignal, action_details: Dict[str, Any]) -> BaseSignal:
        """Create a signal for moderation."""
        enhanced_context = original_signal.context.copy()
        enhanced_context.update({
            'orchestration': action_details,
            'action_type': 'moderate',
            'moderated_at': datetime.now(timezone.utc).isoformat(),
            'processing_stage': 'moderated'
        })
        
        return BaseSignal(
            signal_id=f"mod_{original_signal.signal_id}",
            author=original_signal.author,
            content=original_signal.content,
            context=enhanced_context
        )
    
    def _create_notification_signal(self, original_signal: BaseSignal, action_details: Dict[str, Any]) -> BaseSignal:
        """Create a signal for admin notification."""
        enhanced_context = original_signal.context.copy()
        enhanced_context.update({
            'orchestration': action_details,
            'action_type': 'notify_admin',
            'notification_at': datetime.now(timezone.utc).isoformat(),
            'processing_stage': 'notified'
        })
        
        return BaseSignal(
            signal_id=f"notify_{original_signal.signal_id}",
            author=original_signal.author,
            content=original_signal.content,
            context=enhanced_context
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = self._stats.copy()
        
        if stats['start_time']:
            runtime = datetime.now(timezone.utc) - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()
        
        # Rule statistics
        stats['total_rules'] = len(self._rules)
        stats['enabled_rules'] = len([r for r in self._rules if r.enabled])
        stats['rule_usage'] = {
            rule.name: {
                'usage_count': rule.usage_count,
                'last_used': rule.last_used.isoformat() if rule.last_used else None
            }
            for rule in self._rules
        }
        
        return stats
    
    def is_running(self) -> bool:
        """Check if the orchestrator is currently running."""
        return self._running
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the orchestrator."""
        health = {
            'orchestrator_running': self.is_running(),
            'signal_bus_healthy': signal_bus is not None,
            'rules_loaded': len(self._rules),
            'stats': self.get_stats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        health['healthy'] = (
            health['orchestrator_running'] and
            health['signal_bus_healthy'] and
            health['rules_loaded'] > 0
        )
        
        return health


# Global orchestrator instance
message_orchestrator = MessageOrchestrator()


async def start_orchestrator() -> None:
    """Start the global orchestrator instance."""
    await message_orchestrator.start()


async def stop_orchestrator() -> None:
    """Stop the global orchestrator instance."""
    await message_orchestrator.stop()


if __name__ == "__main__":
    """
    Direct execution for testing the orchestrator.
    """
    import asyncio
    
    async def test_orchestrator():
        """Test the orchestrator with mock signals."""
        print("🔧 Testing Message Orchestrator...")
        
        # Start the orchestrator
        await message_orchestrator.start()
        print(f"✅ Orchestrator started: {message_orchestrator.is_running()}")
        
        # Create test classified signal
        test_signal = DiscordMessage(
            signal_id="test_123",
            author="testuser",
            content="What time is the meeting today?",
            context={
                'classification': {
                    'message_type': 'question',
                    'confidence': 0.9,
                    'toxicity': 0.1
                },
                'classified_at': datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Publish test signal
        await signal_bus.publish("SIGNAL_CLASSIFIED", test_signal)
        print("✅ Test signal published")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check stats
        stats = message_orchestrator.get_stats()
        print(f"✅ Orchestrator stats: {stats}")
        
        # Health check
        health = await message_orchestrator.health_check()
        print(f"✅ Health check: {health}")
        
        # Stop the orchestrator
        await message_orchestrator.stop()
        print("✅ Orchestrator stopped")
        
        print("✅ Orchestrator test completed!")
    
    asyncio.run(test_orchestrator())