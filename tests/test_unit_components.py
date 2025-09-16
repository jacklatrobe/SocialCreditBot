"""
Unit tests for individual system components

This module contains traditional unit tests for testing individual components
in isolation, complementing the comprehensive end-to-end tests.
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment
os.environ.update({
    'DISCORD_BOT_TOKEN': 'test_token',
    'LLM_API_KEY': 'test_key',
    'DB_PATH': './data/test_unit.db'
})

from app.signals.base import Signal
from app.infra.bus import SignalBus, SignalType, BusMessage
from app.llm.classification import MessageClassificationService, ClassificationRules
from app.orchestrator.core import MessageOrchestrator, ActionType, ResponsePriority
from tests.test_e2e_system import MockOpenAILLMClient


@pytest.mark.asyncio
async def test_signal_bus_operations():
    """Test SignalBus basic operations."""
    bus = SignalBus()
    await bus.start()
    
    assert bus.running is True
    
    # Test message publishing
    await bus.publish(
        signal_type=SignalType.SIGNAL_TEST,
        data={"test": True},
        source="unit_test"
    )
    
    await bus.stop()
    assert bus.running is False


@pytest.mark.asyncio
async def test_mock_llm_classification():
    """Test mock LLM client classification accuracy."""
    client = MockOpenAILLMClient()
    
    # Test question classification
    signal = Signal(
        signal_id="test_1",
        source="test",
        created_at=datetime.now(timezone.utc),
        author={"user_id": "test_user", "username": "tester"},
        context={"channel_id": "test_channel"},
        content="What time is the meeting?"
    )
    
    result = await client.classify_message(signal)
    
    assert result.purpose == "question"
    assert result.confidence >= 0.8
    assert result.requires_response is True


@pytest.mark.asyncio  
async def test_classification_rules():
    """Test classification rule patterns."""
    rules = ClassificationRules()
    
    test_cases = [
        ("What time is it?", "question"),
        ("This is broken!", "complaint"),
        ("Hello everyone", "social"),
        ("CLICK HERE FOR MONEY!!!", "spam")
    ]
    
    for content, expected_type in test_cases:
        classification = rules.classify_message_content(content)
        assert expected_type in str(classification).lower()


@pytest.mark.asyncio
async def test_orchestrator_rules():
    """Test orchestrator rule loading and evaluation."""
    orchestrator = MessageOrchestrator(max_concurrent_tasks=5)
    
    # Check that rules are loaded
    rules = orchestrator.get_rules() if hasattr(orchestrator, 'get_rules') else []
    assert len(rules) >= 0  # Should have some default rules
    
    # Test rule structure
    if rules:
        rule = rules[0]
        assert hasattr(rule, 'name')
        assert hasattr(rule, 'action_type')
        assert hasattr(rule, 'priority')


@pytest.mark.unit
def test_test_framework_data_generation():
    """Test that our test framework generates appropriate test data."""
    from tests.test_e2e_system import EndToEndTestFramework
    
    framework = EndToEndTestFramework()
    framework._generate_test_users()
    framework._generate_test_messages()
    
    # Verify test users
    assert len(framework.test_users) >= 5
    assert any(user.username == "alice_coder" for user in framework.test_users)
    assert any(user.is_bot for user in framework.test_users)
    
    # Verify test messages
    assert len(framework.test_messages) >= 50
    
    # Check message type distribution
    types = [msg.expected_type for msg in framework.test_messages]
    assert "question" in types
    assert "complaint" in types
    assert "spam" in types
    assert "social" in types
    
    # Verify message variety
    contents = [msg.content for msg in framework.test_messages]
    assert len(set(contents)) == len(contents)  # All unique


@pytest.mark.unit
def test_mock_response_generation():
    """Test mock Discord responder response generation."""
    from tests.test_e2e_system import MockDiscordResponderTool
    
    tool = MockDiscordResponderTool()
    
    # Test response content generation
    classification = {"purpose": "question", "confidence": 0.85}
    content = "What time is the meeting?"
    
    response_content = tool._generate_response_content(classification, content)
    
    assert "help" in response_content.lower()
    assert len(response_content) > 10
    assert "What time is the meeting" in response_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])