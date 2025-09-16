"""
Signal Bus for internal component communication.
Async message passing using asyncio.Queue with publish/subscribe pattern.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of signals that flow through the system."""
    SIGNAL_INGESTED = "signal.ingested"           # New signal received
    SIGNAL_CLASSIFIED = "signal.classified"       # Signal tagged by nano-classifier  
    OBSERVER_FLAGGED = "observer.flagged"         # Observer identified actionable signal
    ACTION_REQUESTED = "action.requested"         # Orchestrator requests action
    ACTION_COMPLETED = "action.completed"         # Action tool completed work
    ACTION_FAILED = "action.failed"               # Action tool failed
    SYSTEM_SHUTDOWN = "system.shutdown"           # Graceful shutdown signal
    SIGNAL_TEST = "signal.test"                   # Health check test signal


@dataclass
class BusMessage:
    """Message envelope for signal bus communication."""
    signal_type: SignalType
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    message_id: str
    correlation_id: Optional[str] = None  # For tracking related messages


class SignalBusError(Exception):
    """Signal bus operation error."""
    pass


class SignalBus:
    """
    Async message bus for component communication.
    Uses asyncio.Queue for reliable message delivery.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.message_queue: asyncio.Queue = None
        self.subscribers: Dict[SignalType, List[Callable]] = {}
        self.running = False
        self.message_count = 0
        
        # Message routing task
        self._routing_task: Optional[asyncio.Task] = None
        
        logger.info(f"SignalBus initialized with max_queue_size={max_queue_size}")
    
    async def start(self):
        """Start the signal bus."""
        if self.running:
            return
            
        self.message_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.running = True
        
        # Start message routing task
        self._routing_task = asyncio.create_task(self._route_messages())
        
        logger.info("SignalBus started")
    
    async def stop(self):
        """Stop the signal bus gracefully."""
        if not self.running:
            return
            
        self.running = False
        
        # Send shutdown signal
        await self.publish(
            signal_type=SignalType.SYSTEM_SHUTDOWN,
            data={},
            source="signal_bus"
        )
        
        # Wait for routing task to finish
        if self._routing_task:
            try:
                await asyncio.wait_for(self._routing_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("SignalBus routing task did not shutdown cleanly")
                self._routing_task.cancel()
        
        logger.info("SignalBus stopped")
    
    async def publish(self, signal_type: SignalType, data: Dict[str, Any], 
                     source: str, correlation_id: Optional[str] = None) -> bool:
        """
        Publish a message to the bus.
        
        Args:
            signal_type: Type of signal
            data: Message payload
            source: Component that generated the message
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            bool: True if message was queued successfully
        """
        if not self.running or not self.message_queue:
            logger.error("SignalBus not running, cannot publish message")
            return False
        
        # Create message
        message = BusMessage(
            signal_type=signal_type,
            data=data,
            timestamp=datetime.utcnow(),
            source=source,
            message_id=f"{source}_{self.message_count}_{int(datetime.utcnow().timestamp())}",
            correlation_id=correlation_id
        )
        
        try:
            # Try to put message in queue (non-blocking)
            self.message_queue.put_nowait(message)
            self.message_count += 1
            
            logger.debug(f"Published message: {signal_type.value} from {source}")
            return True
            
        except asyncio.QueueFull:
            logger.error(f"SignalBus queue full, dropping message: {signal_type.value}")
            return False
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    def subscribe(self, signal_type: SignalType, 
                 handler: Callable[[BusMessage], Awaitable[None]]):
        """
        Subscribe to a signal type.
        
        Args:
            signal_type: Type of signal to subscribe to
            handler: Async function to handle the message
        """
        if signal_type not in self.subscribers:
            self.subscribers[signal_type] = []
        
        self.subscribers[signal_type].append(handler)
        logger.info(f"Subscribed handler to {signal_type.value}")
    
    def unsubscribe(self, signal_type: SignalType, 
                   handler: Callable[[BusMessage], Awaitable[None]]):
        """Unsubscribe from a signal type."""
        if signal_type in self.subscribers:
            try:
                self.subscribers[signal_type].remove(handler)
                logger.info(f"Unsubscribed handler from {signal_type.value}")
            except ValueError:
                logger.warning(f"Handler not found for {signal_type.value}")
    
    async def _route_messages(self):
        """Internal message routing loop."""
        logger.info("SignalBus message routing started")
        
        while self.running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Handle shutdown signal
                if message.signal_type == SignalType.SYSTEM_SHUTDOWN:
                    logger.info("Received shutdown signal, stopping routing")
                    break
                
                # Route message to subscribers
                await self._deliver_message(message)
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue
                continue
            except Exception as e:
                logger.error(f"Error in message routing: {e}")
                # Continue routing despite errors
        
        logger.info("SignalBus message routing stopped")
    
    async def _deliver_message(self, message: BusMessage):
        """Deliver message to all subscribers."""
        handlers = self.subscribers.get(message.signal_type, [])
        
        if not handlers:
            logger.debug(f"No subscribers for {message.signal_type.value}")
            return
        
        # Deliver to all handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_handler_call(handler, message))
            tasks.append(task)
        
        if tasks:
            # Wait for all handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any handler failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {i} failed for {message.signal_type.value}: {result}")
    
    async def _safe_handler_call(self, handler: Callable, message: BusMessage):
        """Safely call a message handler."""
        try:
            await handler(message)
        except Exception as e:
            logger.error(f"Handler error: {e}")
            # Don't re-raise to prevent disrupting other handlers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal bus statistics."""
        return {
            "running": self.running,
            "message_count": self.message_count,
            "queue_size": self.message_queue.qsize() if self.message_queue else 0,
            "max_queue_size": self.max_queue_size,
            "subscriber_counts": {
                signal_type.value: len(handlers) 
                for signal_type, handlers in self.subscribers.items()
            }
        }


# Global signal bus instance
signal_bus: Optional[SignalBus] = None


async def init_signal_bus(max_queue_size: int = 1000) -> SignalBus:
    """Initialize the global signal bus."""
    global signal_bus
    signal_bus = SignalBus(max_queue_size=max_queue_size)
    await signal_bus.start()
    return signal_bus


def get_signal_bus() -> SignalBus:
    """Get the global signal bus instance."""
    if signal_bus is None:
        raise SignalBusError("Signal bus not initialized")
    return signal_bus


async def publish_signal(signal_type: SignalType, data: Dict[str, Any], 
                        source: str, correlation_id: Optional[str] = None) -> bool:
    """Convenience function to publish to global signal bus."""
    bus = get_signal_bus()
    return await bus.publish(signal_type, data, source, correlation_id)


def subscribe_to_signal(signal_type: SignalType, 
                       handler: Callable[[BusMessage], Awaitable[None]]):
    """Convenience function to subscribe to global signal bus."""
    bus = get_signal_bus()
    bus.subscribe(signal_type, handler)


async def close_signal_bus():
    """Close the global signal bus."""
    global signal_bus
    if signal_bus:
        await signal_bus.stop()
        signal_bus = None