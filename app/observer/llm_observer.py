"""
LLM Observer for Discord Observer/Orchestrator

This module implements the Observer component that:
1. Subscribes to SIGNAL_INGESTED events from the signal bus
2. Performs LLM-based classification of Discord messages
3. Publishes SIGNAL_CLASSIFIED events for orchestrator processing

The Observer integrates the classification system with the signal bus,
providing the bridge between message ingestion and orchestration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.signals import Signal as BaseSignal, DiscordMessage
from app.infra.bus import get_signal_bus, SignalType, BusMessage
from app.llm.client import OpenAILLMClient
from app.llm.classification import MessageClassificationService


logger = logging.getLogger(__name__)


class LLMObserver:
    """
    Observer component that processes ingested Discord messages through LLM classification.
    
    This class acts as the bridge between message ingestion and orchestration by:
    - Subscribing to SIGNAL_INGESTED events
    - Performing LLM classification on Discord messages
    - Publishing SIGNAL_CLASSIFIED events with enhanced message context
    
    The Observer implements the observer pattern for reactive message processing
    and ensures that all ingested messages are properly classified before orchestration.
    """
    
    def __init__(self, llm_client: Optional[OpenAILLMClient] = None):
        """
        Initialize the LLM Observer.
        
        Args:
            llm_client: Optional LLM client. If not provided, creates a new instance.
        """
        self.llm_client = llm_client or OpenAILLMClient()
        self.classification_service = MessageClassificationService(self.llm_client)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats = {
            'messages_processed': 0,
            'classifications_completed': 0,
            'classification_errors': 0,
            'last_processed': None,
            'start_time': None
        }
        
        logger.info("LLM Observer initialized")
    
    async def start(self) -> None:
        """
        Start the LLM Observer to begin processing signals.
        
        Uses proper publish/subscribe pattern by registering a handler
        with the signal bus instead of polling for messages.
        """
        if self._running:
            logger.warning("LLM Observer is already running")
            return
        
        self._running = True
        self._stats['start_time'] = datetime.now(timezone.utc)
        
        # Subscribe to ingested signals using proper event-driven pattern
        signal_bus = get_signal_bus()
        signal_bus.subscribe(SignalType.SIGNAL_INGESTED, self._handle_bus_message)
        
        logger.info("LLM Observer started")
    
    async def stop(self) -> None:
        """
        Stop the LLM Observer and clean up resources.
        
        Since we use event-driven subscription, no background tasks to cancel.
        """
        if not self._running:
            logger.warning("LLM Observer is not running")
            return
        
        self._running = False
        logger.info("LLM Observer stopped")
    
    async def _handle_bus_message(self, message: BusMessage) -> None:
        """
        Handle incoming bus messages containing ingested signals.
        
        This method is called by the signal bus when SIGNAL_INGESTED messages
        are published. Uses proper event-driven architecture.
        
        Args:
            message: Bus message containing the ingested signal
        """
        try:
            if not self._running:
                return  # Ignore messages if observer is stopped
                
            # Extract signal data from bus message
            signal_data = message.data
            
            # Convert to appropriate signal type
            if 'discord_message' in signal_data:
                signal = DiscordMessage(**signal_data['discord_message'])
            else:
                logger.warning(f"Unknown signal type in message: {message.message_id}")
                return
            
            await self._handle_ingested_signal(signal)
            
        except Exception as e:
            logger.error(f"Error handling bus message {message.message_id}: {e}")
            self._stats['classification_errors'] += 1
    
    async def _handle_ingested_signal(self, signal: BaseSignal) -> None:
        """
        Handle an individual ingested signal by performing classification.
        
        Args:
            signal: The ingested signal to process
        """
        try:
            self._stats['messages_processed'] += 1
            self._stats['last_processed'] = datetime.now(timezone.utc)
            
            logger.debug(f"Processing signal {signal.signal_id} from {signal.author}")
            
            # Perform LLM classification
            classification_result = await self.classification_service.classify_message(signal)
            
            if classification_result:
                # Create classified signal with enhanced context
                classified_signal = self._create_classified_signal(signal, classification_result)
                
                # Publish classified signal for orchestrator processing
                signal_bus = get_signal_bus()
                await signal_bus.publish(
                    signal_type=SignalType.SIGNAL_CLASSIFIED,
                    data={'classified_signal': classified_signal.__dict__},
                    source="llm_observer",
                    correlation_id=signal.signal_id
                )
                
                self._stats['classifications_completed'] += 1
                logger.debug(f"Published SIGNAL_CLASSIFIED for {signal.signal_id}")
                
            else:
                logger.warning(f"Classification failed for signal {signal.signal_id}")
                self._stats['classification_errors'] += 1
                
        except Exception as e:
            logger.error(f"Error handling signal {signal.signal_id}: {e}")
            self._stats['classification_errors'] += 1
    
    def _create_classified_signal(self, original_signal: BaseSignal, classification: Dict[str, Any]) -> BaseSignal:
        """
        Create a new signal with classification results added to context.
        
        Args:
            original_signal: The original ingested signal
            classification: Classification results from the LLM service
            
        Returns:
            New signal with classification context added
        """
        # Create enhanced context with classification results
        enhanced_context = original_signal.context.copy()
        enhanced_context.update({
            'classification': classification,
            'classified_at': datetime.now(timezone.utc).isoformat(),
            'classifier_version': 'v1.0',
            'processing_stage': 'classified'
        })
        
        # Create new signal with same data but enhanced context
        if isinstance(original_signal, DiscordMessage):
            classified_signal = DiscordMessage(
                signal_id=original_signal.signal_id,
                author=original_signal.author,
                content=original_signal.content,
                context=enhanced_context
            )
        else:
            # For other signal types, create a generic BaseSignal
            classified_signal = BaseSignal(
                signal_id=original_signal.signal_id,
                author=original_signal.author,
                content=original_signal.content,
                context=enhanced_context
            )
        
        return classified_signal
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current observer statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        stats = self._stats.copy()
        
        # Add calculated fields
        if stats['start_time']:
            runtime = datetime.now(timezone.utc) - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()
        
        stats['success_rate'] = (
            stats['classifications_completed'] / max(stats['messages_processed'], 1)
        )
        
        stats['error_rate'] = (
            stats['classification_errors'] / max(stats['messages_processed'], 1)
        )
        
        return stats
    
    def is_running(self) -> bool:
        """Check if the observer is currently running."""
        return self._running
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the observer and its dependencies.
        
        Returns:
            Health check results
        """
        health = {
            'observer_running': self.is_running(),
            'signal_bus_healthy': signal_bus is not None,
            'stats': self.get_stats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check LLM client health
        try:
            llm_health = await self.llm_client.health_check()
            health['llm_client'] = llm_health
        except Exception as e:
            health['llm_client'] = {'healthy': False, 'error': str(e)}
        
        # Overall health assessment
        health['healthy'] = (
            health['observer_running'] and 
            health['signal_bus_healthy'] and 
            health['llm_client'].get('healthy', False)
        )
        
        return health


# Global observer instance
llm_observer = LLMObserver()


async def start_llm_observer() -> None:
    """Start the global LLM observer instance."""
    await llm_observer.start()


async def stop_llm_observer() -> None:
    """Stop the global LLM observer instance."""
    await llm_observer.stop()


if __name__ == "__main__":
    """
    Direct execution for testing the LLM Observer.
    """
    import asyncio
    
    async def test_observer():
        """Test the LLM Observer with mock signals."""
        print("🔧 Testing LLM Observer...")
        
        # Start the observer
        await llm_observer.start()
        print(f"✅ Observer started: {llm_observer.is_running()}")
        
        # Create a test signal
        test_signal = DiscordMessage.from_discord_message({
            'id': '12345',
            'content': 'Hello everyone! How are you doing today?',
            'author': {'id': '67890', 'username': 'testuser'},
            'channel_id': '11111',
            'guild_id': '22222',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'attachments': [],
            'embeds': [],
            'mentions': []
        })
        
        # Publish test signal
        await signal_bus.publish("SIGNAL_INGESTED", test_signal)
        print("✅ Test signal published")
        
        # Wait a moment for processing
        await asyncio.sleep(2)
        
        # Check stats
        stats = llm_observer.get_stats()
        print(f"✅ Observer stats: {stats}")
        
        # Health check
        health = await llm_observer.health_check()
        print(f"✅ Health check: {health}")
        
        # Stop the observer
        await llm_observer.stop()
        print("✅ Observer stopped")
        
        print("✅ LLM Observer test completed!")
    
    asyncio.run(test_observer())