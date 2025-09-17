"""
ReAct Agent Orchestrator Implementation

This module integrates the LangGraph ReAct agent with the existing orchestrator
infrastructure, preserving signal handling, priority queues, and retry logic
while using an aggregator to determine when to invoke the expensive ReAct agent.
"""

import asyncio
import logging
import json
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
from enum import Enum

from app.signals import Signal as BaseSignal, DiscordMessage
from app.infra.bus import get_signal_bus, SignalType, BusMessage
from app.config import get_settings
from app.orchestrator.react_agent import get_orchestration_graph
from app.orchestrator.react_context import OrchestrationContext
from app.orchestrator.react_state import OrchestrationInput
from app.orchestrator.aggregator import get_aggregator, AggregatedSignal, ActionTrigger


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for orchestrator tasks (preserved from original)."""
    HIGH = 1      # Escalations, errors, urgent responses
    NORMAL = 2    # Standard responses, questions
    LOW = 3       # Social interactions, logging


@dataclass
class PriorityTask:
    """A task with priority for the orchestrator queue (aggregated signals only)."""
    priority: TaskPriority
    timestamp: datetime
    aggregated_signal: AggregatedSignal
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if self.priority.value == other.priority.value:
            return self.timestamp < other.timestamp
        return self.priority.value < other.priority.value


class ReactMessageOrchestrator:
    """
    AI-powered message orchestrator using aggregation + LangGraph ReAct agent.
    
    This orchestrator:
    1. Receives all classified messages and aggregates user behavior
    2. Uses rules engine to determine when ReAct agent should be invoked
    3. Only calls expensive ReAct agent when aggregate signals warrant action
    4. Preserves existing infrastructure (priority queues, retry logic, etc.)
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the ReAct message orchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent ReAct agent tasks
        """
        self.settings = get_settings()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Worker pool management (for ReAct agent tasks only)
        self._max_concurrent_tasks = max_concurrent_tasks
        self._task_semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: set = set()
        
        # Priority queue system (for triggered aggregated signals only)
        self._priority_queue: List[PriorityTask] = []
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._queue_event: Optional[asyncio.Event] = None
        
        # User profile aggregator (NEW - the key missing component)
        self._aggregator = get_aggregator()
        
        # ReAct agent components (only for triggered responses)
        self._agent_graph = get_orchestration_graph()
        self._agent_context = OrchestrationContext()
        
        # Statistics (enhanced for aggregation)
        self._stats = {
            'messages_received': 0,
            'messages_aggregated': 0,
            'rules_triggered': 0,
            'react_agent_invocations': 0,
            'responses_sent': 0,
            'no_action_decisions': 0,
            'errors': 0,
            'start_time': None,
            'last_processed': None,
            'max_concurrent_tasks': max_concurrent_tasks,
            'active_tasks': 0,
            'priority_queue_size': 0,
            'high_priority_processed': 0,
            'normal_priority_processed': 0,
            'low_priority_processed': 0,
            'agent_reasoning_steps': 0,
            'tool_calls_made': 0
        }
        
        logger.info("ReAct Message Orchestrator initialized with user profile aggregation")
    
    async def start(self) -> None:
        """Start the orchestrator and begin processing signals."""
        if self._running:
            logger.warning("Orchestrator is already running")
            return
        
        logger.info(f"Starting ReAct Message Orchestrator with {self._max_concurrent_tasks} max concurrent tasks")
        
        # Initialize components
        self._running = True
        self._task_semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        self._queue_event = asyncio.Event()
        self._stats['start_time'] = datetime.now(timezone.utc)
        
        # Subscribe to classified signals
        signal_bus = get_signal_bus()
        signal_bus.subscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        
        # Start the priority queue processor
        self._queue_processor_task = asyncio.create_task(self._process_priority_queue())
        
        logger.info("Message Orchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        logger.info("Stopping ReAct Message Orchestrator...")
        self._running = False
        
        # Cancel the queue processor
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete
        if self._active_tasks:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        
        # Unsubscribe from signals
        signal_bus = get_signal_bus()
        signal_bus.unsubscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        
        logger.info("Message Orchestrator stopped")
    
    async def _handle_classified_signal(self, message: BusMessage) -> None:
        """
        Handle a classified signal by aggregating user behavior and 
        determining if ReAct agent should be triggered.
        
        Args:
            message: The bus message containing the classified signal
        """
        try:
            self._stats['messages_received'] += 1
            logger.info(f"🔄 ReAct orchestrator received classified signal: {message.message_id}")
            
            # Extract the classified signal data
            signal_data = message.data.get('classified_signal')
            if not signal_data:
                logger.warning(f"No classified_signal data found in message {message.message_id}")
                return
            
            # Reconstruct the signal from the classified signal data
            if isinstance(signal_data, dict):
                try:
                    # For classified signals, check if it's a DiscordMessage by looking at required fields
                    if all(field in signal_data for field in ['author', 'content', 'source', 'created_at']):
                        signal_data = DiscordMessage.from_dict(signal_data)
                    else:
                        # Fall back to BaseSignal
                        signal_data = BaseSignal(**signal_data)
                except Exception as e:
                    logger.error(f"Failed to reconstruct classified signal from dict: {e}")
                    logger.error(f"Signal data keys: {list(signal_data.keys()) if isinstance(signal_data, dict) else 'Not a dict'}")
                    return
            
            if not isinstance(signal_data, BaseSignal):
                logger.warning(f"Invalid signal data type in message {message.message_id}: {type(signal_data)}")
                return
            
            # Aggregate the message into user profile
            aggregated_signal = await self._aggregator.process_classified_message(signal_data)
            self._stats['messages_aggregated'] += 1
            
            # If aggregation didn't trigger any rules, we're done (no expensive ReAct call)
            if not aggregated_signal:
                logger.debug(f"Message {message.message_id} aggregated but no action triggered")
                return
            
            self._stats['rules_triggered'] += 1
            logger.info(f"Aggregation triggered {aggregated_signal.trigger.value} for user {aggregated_signal.user_id}")
            
            # Determine priority for ReAct agent processing
            priority = self._determine_task_priority_from_trigger(aggregated_signal)
            
            # Create priority task for ReAct agent
            priority_task = PriorityTask(
                priority=priority,
                timestamp=datetime.now(timezone.utc),
                aggregated_signal=aggregated_signal
            )
            
            # Add to priority queue for ReAct agent processing
            heappush(self._priority_queue, priority_task)
            self._stats['priority_queue_size'] = len(self._priority_queue)
            
            # Signal the queue processor
            if self._queue_event:
                self._queue_event.set()
                
            logger.debug(f"Queued aggregated signal {aggregated_signal.trigger.value} with {priority.name} priority")
            
        except Exception as e:
            logger.error(f"Error handling classified signal {message.message_id}: {e}")
            self._stats['errors'] += 1
    
    def _determine_task_priority_from_trigger(self, aggregated_signal: AggregatedSignal) -> TaskPriority:
        """
        Determine task priority based on aggregation trigger.
        
        Args:
            aggregated_signal: The aggregated signal from user behavior analysis
            
        Returns:
            TaskPriority for ReAct agent processing
        """
        trigger = aggregated_signal.trigger
        urgency_score = aggregated_signal.urgency_score
        
        # High priority triggers
        if trigger in [ActionTrigger.TOXIC_ESCALATION, ActionTrigger.HIGH_URGENCY]:
            return TaskPriority.HIGH
            
        # Complaints always get normal priority for quick response
        if trigger == ActionTrigger.COMPLAINT:
            return TaskPriority.NORMAL
            
        # Questions from new users get priority
        if trigger == ActionTrigger.NEW_USER_QUESTION:
            return TaskPriority.NORMAL
            
        # Regular questions and repeat issues
        if trigger in [ActionTrigger.QUESTION, ActionTrigger.REPEAT_ISSUE]:
            # Use urgency score to determine priority
            if urgency_score > 0.7:
                return TaskPriority.NORMAL
            else:
                return TaskPriority.LOW
                
        # Default to low priority
        return TaskPriority.LOW
    
    async def _process_priority_queue(self):
        """Process tasks from the priority queue using the worker pool."""
        logger.info("Priority queue processor started")
        
        try:
            while self._running:
                # Wait for tasks or shutdown
                if not self._priority_queue:
                    await self._queue_event.wait()
                    self._queue_event.clear()
                    continue
                
                # Get the highest priority task
                priority_task = heappop(self._priority_queue)
                self._stats['priority_queue_size'] = len(self._priority_queue)
                
                # Update priority stats
                if priority_task.priority == TaskPriority.HIGH:
                    self._stats['high_priority_processed'] += 1
                elif priority_task.priority == TaskPriority.NORMAL:
                    self._stats['normal_priority_processed'] += 1
                else:
                    self._stats['low_priority_processed'] += 1
                
                # Process task with semaphore control
                async with self._task_semaphore:
                    task = asyncio.create_task(self._process_task(priority_task))
                    self._active_tasks.add(task)
                    self._stats['active_tasks'] = len(self._active_tasks)
                    
                    # Clean up completed task
                    task.add_done_callback(self._active_tasks.discard)
                    
        except asyncio.CancelledError:
            logger.info("Priority queue processor cancelled")
        except Exception as e:
            logger.error(f"Error in priority queue processor: {e}")
    
    async def _process_task(self, priority_task: PriorityTask):
        """
        Process a single aggregated signal using the ReAct agent.
        
        Args:
            priority_task: The priority task containing aggregated signal
        """
        try:
            aggregated_signal = priority_task.aggregated_signal
            
            logger.debug(f"Processing ReAct agent task for trigger {aggregated_signal.trigger.value} - user {aggregated_signal.user_id}")
            
            # Use ReAct agent for orchestration decision with full aggregated context
            await self._orchestrate_with_react_agent(aggregated_signal)
            
        except Exception as e:
            logger.error(f"Error processing aggregated signal task: {e}")
            self._stats['errors'] += 1
    
    async def _orchestrate_with_react_agent(self, aggregated_signal: AggregatedSignal) -> None:
        """
        Use the ReAct agent to make orchestration decisions based on aggregated user behavior.
        
        Args:
            aggregated_signal: The aggregated signal containing user profile and trigger context
        """
        try:
            self._stats['react_agent_invocations'] += 1
            self._stats['last_processed'] = datetime.now(timezone.utc)
            
            original_signal = aggregated_signal.original_signal
            user_id = aggregated_signal.user_id
            
            # Parse classification data from original signal
            classification = original_signal.context.get('classification', original_signal.metadata.get('classification', {}))
            
            if isinstance(classification, str):
                try:
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Invalid classification JSON for signal {original_signal.signal_id}")
                    classification = {}
            
            logger.info(f"Running ReAct agent for {aggregated_signal.trigger.value} - user {user_id} (urgency: {aggregated_signal.urgency_score:.2f})")
            
            # Enhance context with aggregation data including conversation history
            enhanced_context = {
                **original_signal.context,
                'aggregation_trigger': aggregated_signal.trigger.value,
                'urgency_score': aggregated_signal.urgency_score,
                'user_profile': aggregated_signal.context['profile_summary'],
                'recent_activity': aggregated_signal.context['recent_activity'],
                'conversation_history': aggregated_signal.context.get('conversation_history', []),  # Add conversation context
                'trigger_reason': f"User behavior triggered {aggregated_signal.trigger.value} rule"
            }
            
            # Prepare input for the ReAct agent
            agent_input = OrchestrationInput(
                signal=original_signal,
                classification=classification,
                context=enhanced_context
            )
            
            logger.info(f"🚀 Starting ReAct agent workflow for signal {original_signal.signal_id}")
            logger.info(f"   Agent context: max_steps={self._agent_context.max_steps}, model={self._agent_context.model}")
            logger.info(f"   Signal type: {type(original_signal).__name__}")
            logger.info(f"   Classification data: {classification}")
            logger.info(f"   Enhanced context keys: {list(enhanced_context.keys())}")
            
            # Set execution context for tools to access
            from app.orchestrator.react_tools import set_execution_context
            set_execution_context(original_signal, enhanced_context)
            logger.info(f"   Execution context set for ReAct agent tools")
            
            # Run the ReAct agent
            result = await self._agent_graph.ainvoke(
                agent_input,
                context=self._agent_context,
                recursion_limit=self._agent_context.max_steps
            )
            
            logger.info(f"✅ ReAct agent workflow completed for signal {original_signal.signal_id}")
            logger.info(f"   Result keys: {list(result.keys())}")
            if result.get('messages'):
                logger.info(f"   Total messages in result: {len(result['messages'])}")
                for i, msg in enumerate(result['messages']):
                    msg_type = type(msg).__name__
                    has_tools = hasattr(msg, 'tool_calls') and bool(msg.tool_calls)
                    tool_count = len(msg.tool_calls) if has_tools else 0
                    content_preview = msg.content[:100] if hasattr(msg, 'content') and msg.content else '<no content>'
                    logger.info(f"     Message {i+1}: {msg_type}, content: {repr(content_preview)}, tool_calls: {tool_count}")
            
            # Process final decision
            final_message = result.get('messages', [])[-1] if result.get('messages') else None
            if final_message:
                logger.info(f"📋 Final agent decision:")
                logger.info(f"   Message type: {type(final_message).__name__}")
                if hasattr(final_message, 'content') and final_message.content:
                    logger.info(f"   Decision content: {repr(final_message.content)}")
                if hasattr(final_message, 'tool_calls') and final_message.tool_calls:
                    logger.info(f"   Tools to execute: {len(final_message.tool_calls)}")
                    for j, tool_call in enumerate(final_message.tool_calls):
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        logger.info(f"     Tool {j+1}: {tool_name} with args {tool_args}")
                else:
                    logger.info(f"   Decision: NO ACTION (no tool calls)")
            
            # Update statistics
            self._stats['react_agent_invocations'] += 1
            
            # Count reasoning steps and tool calls
            tool_calls_made = 0
            if result.get('messages'):
                for msg in result['messages']:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls_made += len(msg.tool_calls)
                        logger.info(f"   Found {len(msg.tool_calls)} tool calls in message")
                        for j, tool_call in enumerate(msg.tool_calls):
                            tool_name = tool_call.get('name', 'unknown')
                            logger.info(f"     Tool call {j+1}: {tool_name}")
                
                self._stats['agent_reasoning_steps'] = len(result['messages'])
                self._stats['tool_calls_made'] += tool_calls_made
                
                logger.info(f"📊 Agent reasoning steps: {len(result['messages'])}, tool calls: {tool_calls_made}")
            
            # Determine if a response was sent by checking for tool calls
            response_sent = tool_calls_made > 0
            
            # Check if the agent sent a response
            if response_sent:
                self._stats['responses_sent'] += 1
                logger.info(f"✅ ReAct agent sent response for {aggregated_signal.trigger.value} to user {user_id}")
                
                # Record the response in the aggregator to reduce future urgency
                await self._aggregator.record_response_sent(user_id, original_signal)
            else:
                self._stats['no_action_decisions'] += 1
                logger.info(f"ℹ️ ReAct agent decided no response needed for {aggregated_signal.trigger.value} - user {user_id}")
                
        except Exception as e:
            logger.error(f"Error in ReAct agent orchestration for {aggregated_signal.trigger.value}: {e}")
            self._stats['errors'] += 1
        finally:
            # Always clear execution context after processing
            from app.orchestrator.react_tools import clear_execution_context
            clear_execution_context()
            logger.info(f"   Execution context cleared after ReAct agent processing")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return self._stats.copy()
    
    def is_running(self) -> bool:
        """Check if the orchestrator is running."""
        return self._running


# Global orchestrator instance
_react_orchestrator: Optional[ReactMessageOrchestrator] = None


async def get_react_orchestrator() -> ReactMessageOrchestrator:
    """Get the global ReAct orchestrator instance."""
    global _react_orchestrator
    if _react_orchestrator is None:
        _react_orchestrator = ReactMessageOrchestrator()
    return _react_orchestrator


async def start_react_orchestrator():
    """Start the ReAct orchestrator."""
    orchestrator = await get_react_orchestrator()
    await orchestrator.start()
    logger.info("ReAct orchestrator started")


async def stop_react_orchestrator():
    """Stop the ReAct orchestrator."""
    global _react_orchestrator
    if _react_orchestrator:
        await _react_orchestrator.stop()
        _react_orchestrator = None
    logger.info("ReAct orchestrator stopped")