"""Test message classification system"""
import asyncio
import sys
import os
from datetime import datetime

sys.path.append(".")

# Set fake environment variables
os.environ["DISCORD_BOT_TOKEN"] = "fake_token_for_testing"
os.environ["LLM_API_KEY"] = "fake_api_key_for_testing"

from app.llm.classification import (
    MessageClassificationService, 
    ClassificationRules,
    MessageType,
    EnhancedClassificationResult
)
from app.signals.discord import DiscordMessage

async def test_classification():
    """Test classification system functionality."""
    print("🔧 Testing classification system...")
    
    try:
        # Create classification service (without LLM client for testing)
        service = MessageClassificationService(llm_client=None)
        print(f"✅ Classification service created: {service is not None}")
        
        # Test classification rules
        rules = ClassificationRules()
        print(f"✅ Classification rules created with {len(rules.compiled_patterns)} pattern categories")
        
        # Create test messages
        test_messages = [
            {
                'content': 'What time is the meeting?',
                'expected_purpose': 'question'
            },
            {
                'content': 'THIS IS SPAM CLICK HERE FOR FREE MONEY!!!',
                'expected_purpose': 'spam'
            },
            {
                'content': 'Hello everyone, how are you doing today?',
                'expected_purpose': 'social'
            },
            {
                'content': 'This feature is broken and not working!',
                'expected_purpose': 'complaint'
            }
        ]
        
        # Test each message
        for i, test_msg in enumerate(test_messages):
            signal = DiscordMessage(
                signal_id=f"test-{i}",
                source="test",
                created_at=datetime.now(),
                author={"user_id": "123", "username": "TestUser"},
                context={"guild_id": "456", "channel_id": "789", "message_id": f"00{i}"},
                content=test_msg['content']
            )
            
            result = await service.classify_message(signal)
            print(f"✅ Message {i+1}: '{test_msg['content'][:30]}...' -> {result.purpose} (confidence: {result.confidence:.2f})")
            
            # Check if classification matches expectation
            if result.purpose == test_msg['expected_purpose']:
                print(f"   ✅ Correct classification!")
            else:
                print(f"   ⚠️ Expected {test_msg['expected_purpose']}, got {result.purpose}")
        
        # Test stats
        stats = service.get_stats()
        print(f"✅ Stats: {stats['total_classifications']} total classifications")
        print(f"   Rule-based: {stats['rule_based_classifications']}")
        print(f"   LLM: {stats['llm_classifications']}")
        
        # Test enhanced result features
        test_signal = DiscordMessage(
            signal_id="enhanced-test",
            source="test",
            created_at=datetime.now(),
            author={"user_id": "456", "username": "TestBot"},
            context={"guild_id": "789", "channel_id": "101", "message_id": "enhanced"},
            content="Can someone help me with this error?",
            channel_name="help-desk"
        )
        
        enhanced_result = await service.classify_message(test_signal)
        print(f"✅ Enhanced result:")
        print(f"   Message type: {enhanced_result.message_type}")
        print(f"   Rule matches: {enhanced_result.rule_matches}")
        print(f"   Context factors: {enhanced_result.context_factors}")
        
        print("✅ Classification system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_classification())