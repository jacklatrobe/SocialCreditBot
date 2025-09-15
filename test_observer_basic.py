"""
Simple test for LLM Observer components
"""

import asyncio
import os
from datetime import datetime, timezone

# Set test environment variables
os.environ['DISCORD_BOT_TOKEN'] = 'test_token'
os.environ['LLM_API_KEY'] = 'test_key'
os.environ['DATABASE_URL'] = 'sqlite:///test_observer.db'

from app.observer.llm_observer import LLMObserver


async def test_observer_basic():
    """Test basic LLM Observer functionality without full integration."""
    print("🔧 Testing LLM Observer Basic Functionality...")
    
    try:
        # Create observer instance
        observer = LLMObserver()
        print(f"✅ Observer created: {type(observer).__name__}")
        
        # Check initial state
        print(f"✅ Initial running state: {observer.is_running()}")
        
        # Get initial stats
        stats = observer.get_stats()
        print(f"✅ Initial stats: {stats}")
        
        # Health check (will fail on LLM client but observer should be OK)
        try:
            health = await observer.health_check()
            print(f"✅ Health check completed: {health['observer_running']}")
        except Exception as e:
            print(f"⚠️ Health check had expected LLM client issues: {str(e)[:100]}...")
        
        print(f"✅ LLM Observer basic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_observer_basic())