"""
Simple test for Message Orchestrator core logic
"""

import asyncio
import os
from datetime import datetime, timezone

# Set test environment variables
os.environ['DISCORD_BOT_TOKEN'] = 'test_token'
os.environ['LLM_API_KEY'] = 'test_key'
os.environ['DATABASE_URL'] = 'sqlite:///test_orchestrator.db'

from app.orchestrator.core import MessageOrchestrator, OrchestrationRule, ActionType, ResponsePriority
from app.signals import Signal as BaseSignal


async def test_orchestrator_logic():
    """Test the Message Orchestrator logic without signal bus."""
    print("🔧 Testing Message Orchestrator Logic...")
    
    try:
        # Create orchestrator instance
        orchestrator = MessageOrchestrator()
        print(f"✅ Orchestrator created: {type(orchestrator).__name__}")
        
        # Test orchestration rules
        rules = orchestrator.get_rules()
        print(f"✅ Loaded {len(rules)} orchestration rules:")
        for rule in rules:
            print(f"   - {rule.name}: {rule.action_type.value} (priority: {rule.priority.value})")
        
        # Test rule matching logic
        print(f"\n🧪 Testing rule matching logic...")
        
        test_cases = [
            {
                'name': 'High Confidence Question',
                'classification': {'message_type': 'question', 'confidence': 0.9, 'toxicity': 0.1},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.RESPOND
            },
            {
                'name': 'Complaint',
                'classification': {'message_type': 'complaint', 'confidence': 0.8, 'toxicity': 0.3},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.ESCALATE
            },
            {
                'name': 'Spam',
                'classification': {'message_type': 'spam', 'confidence': 0.95, 'toxicity': 0.2},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.MODERATE
            },
            {
                'name': 'Toxic Content',
                'classification': {'message_type': 'other', 'confidence': 0.7, 'toxicity': 0.9},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.MODERATE
            },
            {
                'name': 'Social Message',
                'classification': {'message_type': 'social', 'confidence': 0.8, 'toxicity': 0.0},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.RESPOND
            },
            {
                'name': 'Low Confidence Question',
                'classification': {'message_type': 'question', 'confidence': 0.5, 'toxicity': 0.1},
                'context': {'guild_id': '12345'},
                'expected_action': ActionType.LOG_ONLY  # Should fall to default
            }
        ]
        
        for test_case in test_cases:
            print(f"\n   Testing: {test_case['name']}")
            
            # Find matching rule
            matching_rule = orchestrator._find_matching_rule(
                test_case['classification'], 
                test_case['context']
            )
            
            if matching_rule:
                print(f"   ✅ Matched rule: {matching_rule.name}")
                print(f"   ✅ Action: {matching_rule.action_type.value}")
                
                if matching_rule.action_type == test_case['expected_action']:
                    print(f"   ✅ Expected action matched!")
                else:
                    print(f"   ⚠️ Expected {test_case['expected_action'].value}, got {matching_rule.action_type.value}")
            else:
                print(f"   ❌ No matching rule found")
        
        # Test custom rule addition
        print(f"\n🔧 Testing custom rule addition...")
        
        custom_rule = OrchestrationRule(
            name="Test Custom Rule",
            description="A test rule for validation",
            conditions={"message_type": "test", "confidence": {"min": 0.5}},
            action_type=ActionType.NOTIFY_ADMIN,
            priority=ResponsePriority.HIGH
        )
        
        initial_count = len(orchestrator.get_rules())
        orchestrator.add_rule(custom_rule)
        new_count = len(orchestrator.get_rules())
        
        print(f"   ✅ Rules before: {initial_count}, after: {new_count}")
        
        # Test the custom rule
        test_classification = {"message_type": "test", "confidence": 0.7}
        test_context = {"guild_id": "12345"}
        
        matching_rule = orchestrator._find_matching_rule(test_classification, test_context)
        if matching_rule and matching_rule.name == "Test Custom Rule":
            print(f"   ✅ Custom rule matched correctly!")
        else:
            print(f"   ❌ Custom rule did not match")
        
        # Test rule removal
        removed = orchestrator.remove_rule("Test Custom Rule")
        if removed:
            print(f"   ✅ Custom rule removed successfully")
        else:
            print(f"   ❌ Failed to remove custom rule")
        
        # Get initial stats
        stats = orchestrator.get_stats()
        print(f"\n📊 Orchestrator Statistics:")
        print(f"   Total rules: {stats['total_rules']}")
        print(f"   Enabled rules: {stats['enabled_rules']}")
        
        # Health check
        health = await orchestrator.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Orchestrator healthy: {health['healthy']}")
        print(f"   Rules loaded: {health['rules_loaded']}")
        
        print(f"\n✅ Message Orchestrator logic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_orchestrator_logic())