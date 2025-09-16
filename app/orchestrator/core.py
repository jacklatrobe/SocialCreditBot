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
import random
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timezone
from enum import Enum

from app.signals import Signal as BaseSignal, DiscordMessage
from app.infra.bus import get_signal_bus, SignalType, BusMessage
from app.config import get_settings
from app.orchestrator.tools import create_discord_responder_tool, ResponseTemplate


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for orchestrator tasks."""
    HIGH = 1      # Escalations, errors, urgent responses
    NORMAL = 2    # Standard responses, questions
    LOW = 3       # Social interactions, logging


@dataclass
class PriorityTask:
    """A task with priority for the orchestrator queue."""
    priority: TaskPriority
    timestamp: datetime
    message: BusMessage
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if self.priority.value == other.priority.value:
            return self.timestamp < other.timestamp  # FIFO for same priority
        return self.priority.value < other.priority.value  # Higher priority first


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt with exponential backoff and jitter."""
        if attempt <= 0:
            return 0
        
        # Exponential backoff: delay = base_delay * 2^(attempt-1)
        delay = self.base_delay * (2 ** (attempt - 1))
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25% randomization)
        jitter = delay * 0.25 * (random.random() * 2 - 1)  # Random between -25% and +25%
        delay = max(0, delay + jitter)
        
        return delay


async def retry_with_backoff(
    func: Callable,
    *args,
    retry_config: RetryConfig = None,
    retryable_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The async function to retry
        *args: Arguments for the function
        retry_config: Retry configuration
        retryable_exceptions: Tuple of exceptions that should trigger retry
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
        
    Raises:
        The last exception if all retries fail
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(retry_config.max_retries + 1):  # +1 for initial attempt
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == retry_config.max_retries:
                # Last attempt failed, re-raise the exception
                logger.error(f"Function {func.__name__} failed after {retry_config.max_retries + 1} attempts: {e}")
                raise
            
            # Calculate delay and wait
            delay = retry_config.get_delay(attempt + 1)
            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_exception


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
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the message orchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent message processing tasks
        """
        self.settings = get_settings()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._rules: List[OrchestrationRule] = []
        self._action_handlers: Dict[ActionType, Callable] = {}
        
        # Worker pool management
        self._max_concurrent_tasks = max_concurrent_tasks
        self._task_semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: set = set()
        
        # Priority queue system
        self._priority_queue: List[PriorityTask] = []
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._queue_event: Optional[asyncio.Event] = None  # Will be initialized in start()
        
        self._stats = {
            'messages_orchestrated': 0,
            'actions_taken': 0,
            'rules_executed': 0,
            'errors': 0,
            'start_time': None,
            'last_processed': None,
            'max_concurrent_tasks': max_concurrent_tasks,
            'active_tasks': 0,
            'task_queue_full_count': 0,
            'priority_queue_size': 0,
            'high_priority_processed': 0,
            'normal_priority_processed': 0,
            'low_priority_processed': 0,
            'retry_attempts': 0,
            'retry_successes': 0,
            'retry_failures': 0
        }
        
        # Initialize Discord responder tool
        self._discord_responder = create_discord_responder_tool()
        
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
        
        # Initialize task management resources
        self._task_semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        self._active_tasks = set()
        
        # Initialize priority queue system
        self._priority_queue = []
        self._queue_event = asyncio.Event()
        self._queue_processor_task = asyncio.create_task(self._process_priority_queue())
        
        # Subscribe to classified signals
        signal_bus = get_signal_bus()
        signal_bus.subscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        
        logger.info(f"Message Orchestrator started with {self._max_concurrent_tasks} max concurrent tasks")
        logger.info("Subscribed to SIGNAL_CLASSIFIED")
    
    async def stop(self) -> None:
        """
        Stop the orchestrator and clean up resources.
        """
        if not self._running:
            logger.warning("Message Orchestrator is not running")
            return
        
        self._running = False
        
        # Unsubscribe from signals
        try:
            signal_bus = get_signal_bus()
            signal_bus.unsubscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        except Exception as e:
            logger.warning(f"Error unsubscribing from signals: {e}")
        
        # Stop the queue processor
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete (with timeout)
        if self._active_tasks:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks to complete, forcing shutdown")
                # Cancel remaining tasks
                for task in self._active_tasks:
                    if not task.done():
                        task.cancel()
        
        # Clean up resources
        self._task_semaphore = None
        self._active_tasks.clear()
        
        logger.info("Message Orchestrator stopped")
    
    async def _handle_classified_signal(self, message: BusMessage) -> None:
        """
        Handle a classified signal from the signal bus by adding it to the priority queue.
        
        Args:
            message: The bus message containing the classified signal
        """
        try:
            # Extract the signal to determine priority
            signal_data = message.data.get('signal')
            if not signal_data or not isinstance(signal_data, BaseSignal):
                logger.warning(f"Invalid signal data in message {message.message_id}")
                return
            
            # Determine priority based on message classification
            priority = self._determine_task_priority(signal_data)
            
            # Create priority task
            priority_task = PriorityTask(
                priority=priority,
                timestamp=datetime.now(timezone.utc),
                message=message
            )
            
            # Add to priority queue
            heappush(self._priority_queue, priority_task)
            self._stats['priority_queue_size'] = len(self._priority_queue)
            
            # Signal the queue processor
            if self._queue_event:
                self._queue_event.set()
            
            logger.debug(f"Queued message {message.message_id} with {priority.name} priority")
            
        except Exception as e:
            logger.error(f"Error queueing classified signal {message.message_id}: {e}")
            self._stats['errors'] += 1
    
    def _determine_task_priority(self, signal: BaseSignal) -> TaskPriority:
        """
        Determine the priority of a task based on the signal classification.
        
        Args:
            signal: The classified signal
            
        Returns:
            TaskPriority for this signal
        """
        try:
            # Get classification data
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            # Parse classification if it's a string (JSON)
            if isinstance(classification, str):
                try:
                    import json
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    classification = {}
            
            # Determine priority based on classification
            message_type = classification.get('message_type', '').lower()
            toxicity = classification.get('toxicity', 0)
            confidence = classification.get('confidence', 0)
            
            # High priority: spam, toxic content, complaints (need immediate action)
            if message_type in ['spam', 'toxic', 'complaint'] or toxicity > 0.5:
                return TaskPriority.HIGH
            
            # Normal priority: questions, requests (need response)
            elif message_type in ['question', 'request'] and confidence > 0.7:
                return TaskPriority.NORMAL
            
            # Low priority: social interactions, general chat
            else:
                return TaskPriority.LOW
                
        except Exception as e:
            logger.warning(f"Error determining priority for signal {signal.signal_id}: {e}")
            return TaskPriority.NORMAL  # Default to normal priority
    
    async def _process_priority_queue(self):
        """
        Process tasks from the priority queue using the worker pool.
        """
        logger.info("Priority queue processor started")
        
        try:
            while self._running:
                # Wait for tasks to be queued or for shutdown
                if not self._priority_queue:
                    await self._queue_event.wait()
                    self._queue_event.clear()
                    
                    # Check if we're shutting down
                    if not self._running:
                        break
                
                # Process tasks while queue has items and workers are available
                while self._priority_queue and self._running:
                    # Check if we have available workers
                    if not self._task_semaphore or self._task_semaphore.locked():
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Get highest priority task
                    priority_task = heappop(self._priority_queue)
                    self._stats['priority_queue_size'] = len(self._priority_queue)
                    
                    # Update priority statistics
                    if priority_task.priority == TaskPriority.HIGH:
                        self._stats['high_priority_processed'] += 1
                    elif priority_task.priority == TaskPriority.NORMAL:
                        self._stats['normal_priority_processed'] += 1
                    else:
                        self._stats['low_priority_processed'] += 1
                    
                    # Create task for processing this message with worker pool management
                    task = asyncio.create_task(
                        self._process_classified_signal_with_semaphore(priority_task.message)
                    )
                    self._active_tasks.add(task)
                    
                    # Clean up completed tasks
                    task.add_done_callback(self._active_tasks.discard)
                    
                    logger.debug(f"Processing {priority_task.priority.name} priority message {priority_task.message.message_id}")
                    
        except asyncio.CancelledError:
            logger.info("Priority queue processor cancelled")
        except Exception as e:
            logger.error(f"Error in priority queue processor: {e}")
        finally:
            logger.info("Priority queue processor stopped")
    
    async def _retry_with_stats(
        self,
        func: Callable,
        *args,
        retry_config: RetryConfig = None,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff and update orchestrator stats.
        
        Args:
            func: The async function to retry
            *args: Arguments for the function
            retry_config: Retry configuration
            retryable_exceptions: Tuple of exceptions that should trigger retry
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            The last exception if all retries fail
        """
        if retry_config is None:
            retry_config = RetryConfig()
        
        last_exception = None
        
        for attempt in range(retry_config.max_retries + 1):  # +1 for initial attempt
            try:
                result = await func(*args, **kwargs)
                
                # Update stats on success
                if attempt > 0:  # Only count as retry success if we actually retried
                    self._stats['retry_successes'] += 1
                    logger.debug(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                self._stats['retry_attempts'] += 1
                
                if attempt == retry_config.max_retries:
                    # Last attempt failed, update failure stats
                    self._stats['retry_failures'] += 1
                    logger.error(f"Function {func.__name__} failed after {retry_config.max_retries + 1} attempts: {e}")
                    raise
                
                # Calculate delay and wait
                delay = retry_config.get_delay(attempt + 1)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    async def _process_classified_signal_with_semaphore(self, message: BusMessage) -> None:
        """
        Process a classified signal with semaphore-based worker pool management.
        
        Args:
            message: The bus message containing the classified signal
        """
        async with self._task_semaphore:
            try:
                self._stats['active_tasks'] = self._max_concurrent_tasks - self._task_semaphore._value
                
                # Extract the signal from the message data
                signal_data = message.data.get('signal')
                if not signal_data or not isinstance(signal_data, BaseSignal):
                    logger.warning(f"Invalid signal data in message {message.message_id}")
                    return
                
                # Process the classified message
                await self._orchestrate_message(signal_data)
                
            except Exception as e:
                logger.error(f"Error handling classified signal {message.message_id}: {e}")
                self._stats['errors'] += 1
    
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
            
            # Parse classification if it's a string (JSON)
            if isinstance(classification, str):
                try:
                    import json
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Invalid classification JSON for signal {signal.signal_id}: {classification}")
                    classification = {}
            
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
        """Handle response generation using DiscordResponderTool with retry logic."""
        logger.info(f"Generating response for signal {signal.signal_id}")
        
        try:
            # Determine appropriate response template based on action details
            template = self._map_template(action_details.get('response_template'))
            
            # Define retry configuration for Discord API calls
            discord_retry_config = RetryConfig(
                max_retries=3,
                base_delay=2.0,  # Start with 2 seconds
                max_delay=30.0   # Max 30 seconds between retries
            )
            
            # Use retry logic for Discord API calls (rate limits, network issues)
            result = await self._retry_with_stats(
                self._send_discord_response,
                signal,
                template,
                retry_config=discord_retry_config,
                retryable_exceptions=(Exception,)  # Retry on any exception
            )
            
            if result['success']:
                logger.info(f"Response sent successfully for signal {signal.signal_id}: "
                          f"Discord message ID {result.get('discord_message_id')}")
            else:
                logger.error(f"Failed to send response for signal {signal.signal_id}: "
                           f"{result.get('error')}")
                    
        except Exception as e:
            logger.error(f"Error in response handler for signal {signal.signal_id}: {e}")
    
    async def _send_discord_response(
        self, 
        signal: BaseSignal, 
        template: Optional[ResponseTemplate]
    ) -> Dict[str, Any]:
        """Send Discord response with the responder tool."""
        async with self._discord_responder as responder:
            return await responder.run(
                signal=signal,
                text=None,  # Let the tool generate from template
                template=template
            )
    
    def _map_template(self, response_template: Optional[str]) -> Optional[ResponseTemplate]:
        """Map orchestration response template to Discord responder template."""
        if not response_template:
            return None
            
        template_mapping = {
            'helpful_response': ResponseTemplate.HELPFUL_RESPONSE,
            'social_response': ResponseTemplate.SOCIAL_RESPONSE,
            'problem_acknowledgment': ResponseTemplate.PROBLEM_ACKNOWLEDGMENT,
            'escalation_notice': ResponseTemplate.ESCALATION_NOTICE,
            'generic_help': ResponseTemplate.GENERIC_HELP
        }
        
        return template_mapping.get(response_template)
    
    async def _handle_escalate(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle escalation to human moderators with retry logic."""
        logger.info(f"Escalating signal {signal.signal_id}")
        
        try:
            # Define retry configuration for escalation (more aggressive since it's urgent)
            escalation_retry_config = RetryConfig(
                max_retries=5,    # More retries for escalations
                base_delay=1.0,   # Start with 1 second
                max_delay=20.0    # Max 20 seconds between retries
            )
            
            # Send escalation notice with retry logic
            result = await self._retry_with_stats(
                self._send_escalation_notice,
                signal,
                retry_config=escalation_retry_config,
                retryable_exceptions=(Exception,)
            )
            
            if result['success']:
                logger.info(f"Escalation notice sent for signal {signal.signal_id}")
            else:
                logger.error(f"Failed to send escalation notice for signal {signal.signal_id}: "
                           f"{result.get('error')}")
            
            # Also create escalation signal for other systems (with retry for signal bus)
            await self._retry_with_stats(
                self._publish_escalation_signal,
                signal,
                action_details,
                retry_config=RetryConfig(max_retries=2, base_delay=0.5),
                retryable_exceptions=(Exception,)
            )
            
        except Exception as e:
            logger.error(f"Error in escalation handler for signal {signal.signal_id}: {e}")
    
    async def _send_escalation_notice(self, signal: BaseSignal) -> Dict[str, Any]:
        """Send escalation notice using Discord responder tool."""
        async with self._discord_responder as responder:
            return await responder.run(
                signal=signal,
                text=None,  # Let the tool generate from template
                template=ResponseTemplate.ESCALATION_NOTICE
            )
    
    async def _publish_escalation_signal(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Publish escalation signal to the signal bus."""
        signal_bus = get_signal_bus()
        escalation_signal = self._create_escalation_signal(signal, action_details)
        await signal_bus.publish(
            SignalType.ACTION_REQUESTED, 
            {'action': 'escalate', 'signal': escalation_signal}, 
            'orchestrator'
        )
    
    async def _handle_moderate(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle moderation actions."""
        logger.info(f"Moderating signal {signal.signal_id}")
        
        # Create moderation signal
        signal_bus = get_signal_bus()
        moderation_signal = self._create_moderation_signal(signal, action_details)
        await signal_bus.publish(
            SignalType.ACTION_REQUESTED,
            {'action': 'moderate', 'signal': moderation_signal},
            'orchestrator'
        )
    
    async def _handle_log_only(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle logging without further action."""
        logger.info(f"Logging signal {signal.signal_id} for analysis")
        
        # Could save to database or analytics system here
        # For now, just ensure it's logged
    
    async def _handle_notify_admin(self, signal: BaseSignal, action_details: Dict[str, Any]) -> None:
        """Handle admin notification."""
        logger.info(f"Notifying admin about signal {signal.signal_id}")
        
        # Create admin notification signal
        signal_bus = get_signal_bus()
        notification_signal = self._create_notification_signal(signal, action_details)
        await signal_bus.publish(
            SignalType.ACTION_REQUESTED,
            {'action': 'notify_admin', 'signal': notification_signal},
            'orchestrator'
        )
    
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
        signal_bus = get_signal_bus()
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
message_orchestrator = MessageOrchestrator(max_concurrent_tasks=10)


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