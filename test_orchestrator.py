"""
Test script for Message Orchestrator
"""

import asyncio
import json
import os
from datetime import datetime, timezone

# Set test environment variables
os.environ['DISCORD_BOT_TOKEN'] = 'test_token'
os.environ['LLM_API_KEY'] = 'test_key'
os.environ['DATABASE_URL'] = 'sqlite:///test_orchestrator.db'

from app.orchestrator.core import message_orchestrator, ActionType, ResponsePriority
from app.signals import Signal as BaseSignal
from app.infra.bus import init_signal_bus, get_signal_bus, SignalType


async def test_orchestrator():
    """Test the Message Orchestrator with mock classified messages."""
    print("🔧 Testing Message Orchestrator...")
    
    # Initialize signal bus first
    signal_bus = await init_signal_bus()
    
    try:
        # Start the orchestrator
        await message_orchestrator.start()
        print(f"✅ Orchestrator started: {message_orchestrator.is_running()}")
        
        # Test orchestration rules
        rules = message_orchestrator.get_rules()
        print(f"✅ Loaded {len(rules)} orchestration rules:")
        for rule in rules:
            print(f"   - {rule.name}: {rule.action_type.value}")
        
        # Create test classified messages
        test_messages = [
            {
                'signal_id': 'test_question_123',
                'author': 'user1',
                'content': 'What time is the meeting today?',
                'classification': {
                    'message_type': 'question',
                    'confidence': 0.9,
                    'toxicity': 0.1
                }
            },
            {
                'signal_id': 'test_complaint_456',
                'author': 'user2',
                'content': 'This feature is completely broken!',
                'classification': {
                    'message_type': 'complaint',
                    'confidence': 0.8,
                    'toxicity': 0.3
                }
            },
            {
                'signal_id': 'test_spam_789',
                'author': 'spammer',
                'content': 'CLICK HERE FOR FREE MONEY!!!',
                'classification': {
                    'message_type': 'spam',
                    'confidence': 0.95,
                    'toxicity': 0.2
                }
            },
            {
                'signal_id': 'test_social_101',
                'author': 'friendlyuser',
                'content': 'Hello everyone! Hope you are all having a great day!',
                'classification': {
                    'message_type': 'social',
                    'confidence': 0.85,
                    'toxicity': 0.0
                }
            }
        ]
        
        # Process test messages
        for i, msg_data in enumerate(test_messages, 1):
            print(f"\n📨 Processing test message {i}: '{msg_data['content'][:50]}...'")
            
            # Create classified signal
            signal = BaseSignal(
                signal_id=msg_data['signal_id'],
                source="discord",
                created_at=datetime.now(timezone.utc),
                author={
                    "user_id": "12345",
                    "username": msg_data['author']
                },
                content=msg_data['content'],
                context={
                    'guild_id': '11111',
                    'channel_id': '67890',
                    'classification': json.dumps(msg_data['classification']),  # Store as JSON string
                    'classified_at': datetime.now(timezone.utc).isoformat(),
                    'classifier_version': 'v1.0',
                    'processing_stage': 'classified'
                }
            )
            
            # Store classification in metadata for orchestrator to access
            signal.metadata['classification'] = msg_data['classification']
            
            # Publish as classified signal
            await signal_bus.publish(
                SignalType.SIGNAL_CLASSIFIED,
                {'signal': signal},
                'test'
            )
            print(f"   ✅ Published SIGNAL_CLASSIFIED")
            
            # Wait for processing
            await asyncio.sleep(0.5)
        
        # Wait for all messages to be processed
        print("\n⏳ Waiting for processing to complete...")
        await asyncio.sleep(2)
        
        # Check orchestrator stats
        stats = message_orchestrator.get_stats()
        print(f"\n📊 Orchestrator Statistics:")
        print(f"   Messages orchestrated: {stats['messages_orchestrated']}")
        print(f"   Actions taken: {stats['actions_taken']}")
        print(f"   Rules executed: {stats['rules_executed']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Total rules: {stats['total_rules']}")
        print(f"   Enabled rules: {stats['enabled_rules']}")
        
        # Show rule usage
        print(f"\n📋 Rule Usage:")
        for rule_name, usage in stats['rule_usage'].items():
            if usage['usage_count'] > 0:
                print(f"   - {rule_name}: {usage['usage_count']} times")
        
        # Health check
        health = await message_orchestrator.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Orchestrator healthy: {health['healthy']}")
        print(f"   Orchestrator running: {health['orchestrator_running']}")
        print(f"   Signal bus healthy: {health['signal_bus_healthy']}")
        print(f"   Rules loaded: {health['rules_loaded']}")
        
        # Test signal subscription (check if response signals are published)
        print(f"\n🔍 Checking for response signals...")
        try:
            # Try to get a response signal (with timeout)
            # Note: The new signal bus uses async generators, so we'll check differently
            # For now, just simulate checking
            print(f"   ℹ️ Response signal checking not implemented for new bus API yet")
        except Exception as e:
            print(f"   ⚠️ Error checking signals: {e}")
        
        print(f"\n✅ Message Orchestrator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        await message_orchestrator.stop()
        await signal_bus.stop()
        print(f"✅ Orchestrator and signal bus stopped")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())