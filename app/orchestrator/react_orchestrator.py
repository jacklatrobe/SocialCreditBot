"""
ReAct Agent Orchestrator Implementation

This module integrates the LangGraph ReAct agent with the existing orchestrator
infrastructure, preserving signal handling, priority queues, and retry logic
while replacing rule-based decisions with AI agent reasoning.
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


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for orchestrator tasks (preserved from original)."""
    HIGH = 1      # Escalations, errors, urgent responses
    NORMAL = 2    # Standard responses, questions
    LOW = 3       # Social interactions, logging


@dataclass
class PriorityTask:
    """A task with priority for the orchestrator queue (preserved from original)."""
    priority: TaskPriority
    timestamp: datetime
    message: BusMessage
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if self.priority.value == other.priority.value:
            return self.timestamp < other.timestamp
        return self.priority.value < other.priority.value


class ReactMessageOrchestrator:
    """
    AI-powered message orchestrator using LangGraph ReAct agent.
    
    This orchestrator preserves all the existing infrastructure (signal handling,
    priority queues, retry logic) but replaces rule-based decision-making with
    an AI agent that can reason about Discord messages and decide responses.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the ReAct message orchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent message processing tasks
        """
        self.settings = get_settings()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Worker pool management (preserved from original)
        self._max_concurrent_tasks = max_concurrent_tasks
        self._task_semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: set = set()
        
        # Priority queue system (preserved from original)
        self._priority_queue: List[PriorityTask] = []
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._queue_event: Optional[asyncio.Event] = None
        
        # ReAct agent components
        self._agent_graph = get_orchestration_graph()
        self._agent_context = OrchestrationContext()
        
        # Statistics (enhanced for ReAct agent)
        self._stats = {
            'messages_orchestrated': 0,
            'agent_decisions': 0,
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
        
        logger.info("ReAct Message Orchestrator initialized")
    
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
        await signal_bus.subscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        
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
        await signal_bus.unsubscribe(SignalType.SIGNAL_CLASSIFIED, self._handle_classified_signal)
        
        logger.info("Message Orchestrator stopped")
    
    async def _handle_classified_signal(self, message: BusMessage) -> None:
        """
        Handle a classified signal by adding it to the priority queue.
        
        Args:
            message: The bus message containing the classified signal
        """
        try:
            # Extract the signal
            signal_data = message.data.get('signal')
            if not signal_data or not isinstance(signal_data, BaseSignal):
                logger.warning(f"Invalid signal data in message {message.message_id}")
                return
            
            # Determine priority (preserved logic from original)
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
        Determine task priority based on signal classification (preserved from original).
        
        Args:
            signal: The classified signal
            
        Returns:
            TaskPriority for this signal
        """
        try:
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            if isinstance(classification, str):
                try:
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    classification = {}
            
            # Priority logic (preserved from original)
            message_type = classification.get('message_type', '').lower()
            toxicity = classification.get('toxicity', 0)
            confidence = classification.get('confidence', 0)
            
            # High priority: spam, toxic content, complaints
            if message_type in ['spam', 'toxic', 'complaint'] or toxicity > 0.5:
                return TaskPriority.HIGH
            
            # Normal priority: questions, requests
            elif message_type in ['question', 'request'] and confidence > 0.7:
                return TaskPriority.NORMAL
            
            # Low priority: social interactions
            else:
                return TaskPriority.LOW
                
        except Exception as e:
            logger.warning(f"Error determining priority for signal {signal.signal_id}: {e}")
            return TaskPriority.NORMAL
    
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
        Process a single orchestration task using the ReAct agent.
        
        Args:
            priority_task: The task to process
        """
        try:
            signal_data = priority_task.message.data.get('signal')
            if not isinstance(signal_data, BaseSignal):
                logger.error(f"Invalid signal in task processing")
                return
            
            # Use ReAct agent for orchestration decision
            await self._orchestrate_with_react_agent(signal_data)
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            self._stats['errors'] += 1
        finally:
            self._stats['active_tasks'] = len(self._active_tasks) - 1
    
    async def _orchestrate_with_react_agent(self, signal: BaseSignal) -> None:
        """
        Use the ReAct agent to make orchestration decisions.
        
        Args:
            signal: The classified signal to orchestrate
        """
        try:
            self._stats['messages_orchestrated'] += 1
            self._stats['last_processed'] = datetime.now(timezone.utc)
            
            # Parse classification data
            classification = signal.context.get('classification', signal.metadata.get('classification', {}))
            
            if isinstance(classification, str):
                try:
                    classification = json.loads(classification)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Invalid classification JSON for signal {signal.signal_id}")
                    classification = {}
            
            logger.debug(f"Running ReAct agent for signal {signal.signal_id}")
            
            # Prepare input for the ReAct agent
            agent_input = OrchestrationInput(
                signal=signal,
                classification=classification,
                context=signal.context
            )
            
            # Run the ReAct agent
            result = await self._agent_graph.ainvoke(
                agent_input,
                context=self._agent_context,
                recursion_limit=self._agent_context.max_steps
            )
            
            # Update statistics
            self._stats['agent_decisions'] += 1
            
            # Count reasoning steps and tool calls
            if result.get('messages'):
                for msg in result['messages']:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        self._stats['tool_calls_made'] += len(msg.tool_calls)
                
                self._stats['agent_reasoning_steps'] = len(result['messages'])
            
            # Check if the agent sent a response
            if result.get('response_sent'):
                self._stats['responses_sent'] += 1
                logger.info(f"ReAct agent sent response for signal {signal.signal_id}")
            else:
                self._stats['no_action_decisions'] += 1
                logger.debug(f"ReAct agent decided no response needed for signal {signal.signal_id}")
            
        except Exception as e:
            logger.error(f"Error in ReAct agent orchestration for signal {signal.signal_id}: {e}")
            self._stats['errors'] += 1
    
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