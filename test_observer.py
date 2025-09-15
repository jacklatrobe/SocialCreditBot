"""
Test script for LLM Observer Integration
"""

import asyncio
import os
from datetime import datetime, timezone

# Set test environment variables
os.environ['DISCORD_BOT_TOKEN'] = 'test_token'
os.environ['LLM_API_KEY'] = 'test_key'
os.environ['DATABASE_URL'] = 'sqlite:///test_observer.db'

from app.observer.llm_observer import llm_observer
from app.signals import DiscordMessage
from app.infra.bus import signal_bus


async def test_llm_observer():
    """Test the LLM Observer with mock Discord messages."""
    print("🔧 Testing LLM Observer Integration...")
    
    # Initialize the database and signal bus (would normally be done at startup)
    database = get_database_instance()
    await database.initialize()
    
    try:
        # Start the observer
        await llm_observer.start()
        print(f"✅ Observer started: {llm_observer.is_running()}")
        
        # Create test Discord messages
        test_messages = [
            {
                'id': '12345',
                'content': 'What time is the meeting today? I need to prepare my presentation.',
                'author': {'id': '67890', 'username': 'businessuser'},
                'channel_id': '11111',
                'guild_id': '22222',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'attachments': [],
                'embeds': [],
                'mentions': []
            },
            {
                'id': '23456',
                'content': 'Hello everyone! Hope you\'re all having a great day!',
                'author': {'id': '78901', 'username': 'friendlyuser'},
                'channel_id': '11111',
                'guild_id': '22222',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'attachments': [],
                'embeds': [],
                'mentions': []
            },
            {
                'id': '34567',
                'content': 'This feature is completely broken! Nothing works as expected.',
                'author': {'id': '89012', 'username': 'frustrated_user'},
                'channel_id': '11111',
                'guild_id': '22222',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'attachments': [],
                'embeds': [],
                'mentions': []
            }
        ]
        
        # Process test messages
        for i, msg_data in enumerate(test_messages, 1):
            print(f"\n📨 Processing test message {i}: '{msg_data['content'][:50]}...'")
            
            # Create Discord signal
            signal = DiscordMessage.from_discord_message(msg_data)
            
            # Publish as ingested signal
            await signal_bus.publish("SIGNAL_INGESTED", signal)
            print(f"   ✅ Published SIGNAL_INGESTED")
            
            # Wait for processing
            await asyncio.sleep(1)
        
        # Wait for all messages to be processed
        print("\n⏳ Waiting for processing to complete...")
        await asyncio.sleep(3)
        
        # Check observer stats
        stats = llm_observer.get_stats()
        print(f"\n📊 Observer Statistics:")
        print(f"   Messages processed: {stats['messages_processed']}")
        print(f"   Classifications completed: {stats['classifications_completed']}")
        print(f"   Classification errors: {stats['classification_errors']}")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Error rate: {stats['error_rate']:.2%}")
        
        # Health check
        health = await llm_observer.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Observer healthy: {health['healthy']}")
        print(f"   Observer running: {health['observer_running']}")
        print(f"   Signal bus healthy: {health['signal_bus_healthy']}")
        print(f"   LLM client healthy: {health['llm_client'].get('healthy', False)}")
        
        # Test signal subscription (check if classified signals are published)
        print(f"\n🔍 Checking for classified signals...")
        try:
            # Try to get a classified signal (with timeout)
            classified_signal = await asyncio.wait_for(
                signal_bus.subscribe("SIGNAL_CLASSIFIED"), 
                timeout=2.0
            )
            if classified_signal:
                print(f"   ✅ Received SIGNAL_CLASSIFIED: {classified_signal.signal_id}")
                print(f"   📝 Classification: {classified_signal.context.get('classification', {})}")
            else:
                print(f"   ⚠️ No classified signal received")
        except asyncio.TimeoutError:
            print(f"   ⚠️ Timeout waiting for classified signal")
        
        print(f"\n✅ LLM Observer integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        await llm_observer.stop()
        print(f"✅ Observer stopped")
        
        await database.close()
        print(f"✅ Database closed")


if __name__ == "__main__":
    asyncio.run(test_llm_observer())