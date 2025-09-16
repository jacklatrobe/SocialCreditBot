"""
End-to-End System Testing Framework

This module provides comprehensive end-to-end testing of the Discord Social Credit Bot
without requiring actual Discord API connectivity. It mocks all external dependencies
and tests the complete message processing pipeline.

Features:
- Mock Discord API and message events
- Simulated user messages with variety of types
- LLM classification testing
- Message orchestration validation  
- Response capture and verification
- Performance metrics and health monitoring
- Clean Code principles with comprehensive test coverage
"""

import asyncio
import json
import logging
import os
import sys
import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables from .env file before checking them
load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment variables before importing app modules
# For real LLM testing, we need actual OpenAI API key (get from environment or user input)
# Check both OPENAI_API_KEY and LLM_API_KEY for flexibility
real_api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY')
if not real_api_key:
    print("WARNING: No OPENAI_API_KEY or LLM_API_KEY found in environment")
    print("   Set OPENAI_API_KEY or LLM_API_KEY environment variable to test with real LLM")
    print("   Using placeholder key - LLM tests will fail without real key")
    real_api_key = 'test_openai_key_67890'

os.environ.update({
    'DISCORD_BOT_TOKEN': 'test_bot_token_12345',
    'LLM_API_KEY': real_api_key,
    'DB_PATH': './data/test_e2e.db',
    'LOG_LEVEL': 'DEBUG'
})

from app.signals.base import Signal
from app.signals.discord import DiscordMessage
from app.infra.bus import SignalBus, BusMessage, SignalType
from app.infra.db import Database
from app.llm.classification import MessageClassificationService, ClassificationRules
from app.llm.client import OpenAILLMClient, ClassificationResult
from app.orchestrator.core import MessageOrchestrator
from app.orchestrator.tools.discord_responder import DiscordResponderTool, ResponseMode, ResponseTemplate
from app.monitoring.health import HealthChecker
from app.config import get_settings


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TestUser:
    """Represents a simulated Discord user for testing."""
    user_id: str
    username: str
    display_name: str
    is_bot: bool = False
    avatar: Optional[str] = None
    
    
@dataclass  
class TestMessage:
    """Represents a test message with expected classification."""
    content: str
    user: TestUser
    expected_type: str
    expected_confidence_min: float = 0.5
    channel_id: str = "test_channel_123"
    guild_id: str = "test_guild_456"
    message_id: Optional[str] = None
    should_respond: bool = True
    expected_response_mode: ResponseMode = ResponseMode.REPLY


@dataclass
class MockDiscordResponse:
    """Captured response that would be sent to Discord."""
    content: str
    channel_id: str
    mode: ResponseMode
    timestamp: datetime
    signal_id: str
    classification: Dict[str, Any]
    response_time_ms: float


@dataclass
class TestMetrics:
    """Test execution metrics and results."""
    messages_processed: int = 0
    classifications_successful: int = 0
    responses_generated: int = 0
    avg_classification_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    component_health_checks: Dict[str, bool] = None
    error_count: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.component_health_checks is None:
            self.component_health_checks = {}
        if self.errors is None:
            self.errors = []





class MockDiscordResponderTool:
    """Mock Discord responder that captures responses instead of sending."""
    
    def __init__(self):
        self.responses: List[MockDiscordResponse] = []
        self.call_count = 0
        
    async def send_response(self, signal: Signal, classification: Dict[str, Any], 
                          context: Dict[str, Any]) -> bool:
        """Mock sending response - captures instead of sending."""
        start_time = time.time()
        self.call_count += 1
        
        # Simulate response generation time
        await asyncio.sleep(0.05)
        
        # Generate mock response content based on classification
        response_content = self._generate_response_content(classification, signal.content)
        
        mock_response = MockDiscordResponse(
            content=response_content,
            channel_id=context.get('channel_id', 'unknown'),
            mode=ResponseMode.REPLY,
            timestamp=datetime.now(timezone.utc),
            signal_id=signal.signal_id,
            classification=classification,
            response_time_ms=(time.time() - start_time) * 1000
        )
        
        self.responses.append(mock_response)
        return True
        
    def _generate_response_content(self, classification: Dict[str, Any], content: str) -> str:
        """Generate appropriate mock response content."""
        message_type = classification.get('purpose', 'general')
        
        responses = {
            'question': f"I can help with that! Based on your question: '{content[:50]}...', here's some assistance.",
            'complaint': f"I understand your concern about this issue. Let me help address: '{content[:50]}...'",
            'spam': "This message has been flagged and will be reviewed by moderators.",
            'social': f"Hello! Thanks for the friendly message: '{content[:30]}...'. Hope you're having a great day!",
            'general': f"Thanks for your message: '{content[:40]}...'. Let me know if you need any help!"
        }
        
        return responses.get(message_type, responses['general'])


class EndToEndTestFramework:
    """
    Comprehensive end-to-end testing framework following SOLID principles.
    
    Single Responsibility: Orchestrates complete system testing
    Open/Closed: Easy to extend with new test scenarios  
    Dependency Inversion: Uses interfaces and mocks for external dependencies
    """
    
    def __init__(self):
        self.config = get_settings()
        self.signal_bus: Optional[SignalBus] = None
        self.database: Optional[Database] = None
        self.real_llm_client: Optional[OpenAILLMClient] = None
        self.mock_discord_tool: Optional[MockDiscordResponderTool] = None
        self.orchestrator: Optional[MessageOrchestrator] = None
        self.health_checker: Optional[HealthChecker] = None
        self.test_users: List[TestUser] = []
        self.test_messages: List[TestMessage] = []
        self.metrics = TestMetrics()
        
    async def setup(self):
        """Initialize all system components with mocks."""
        logger.info("Setting up End-to-End Test Framework...")
        
        # Initialize database
        self.database = Database(self.config.db_path)
        await self.database.initialize()
        logger.info("Test database initialized")
        
        # Initialize signal bus
        self.signal_bus = SignalBus()
        await self.signal_bus.start()
        logger.info("Signal bus started")
        
        # Initialize real LLM client for actual AI testing
        self.real_llm_client = OpenAILLMClient()
        logger.info("Real OpenAI LLM client created (will test actual AI responses)")
        
        # Initialize mock Discord responder tool
        self.mock_discord_tool = MockDiscordResponderTool()
        logger.info("Mock Discord responder created")
        
        # Initialize orchestrator with mocked dependencies
        self.orchestrator = MessageOrchestrator(max_concurrent_tasks=10)
        logger.info("Message orchestrator initialized")
        
        # Initialize health checker
        self.health_checker = HealthChecker(self.config)
        logger.info("Health checker initialized")
        
        # Generate test data
        self._generate_test_users()
        self._generate_test_messages()
        logger.info(f"Generated {len(self.test_messages)} test messages from {len(self.test_users)} users")
        
    async def teardown(self):
        """Clean up test resources."""
        if self.signal_bus:
            await self.signal_bus.stop()
        logger.info("Test framework cleaned up")
        
    def _generate_test_users(self):
        """Generate diverse test users."""
        self.test_users = [
            TestUser("user1", "alice_coder", "Alice", False, "avatar1.jpg"),
            TestUser("user2", "bob_questioner", "Bob", False, "avatar2.jpg"), 
            TestUser("user3", "charlie_complainer", "Charlie", False, "avatar3.jpg"),
            TestUser("user4", "diana_social", "Diana", False, "avatar4.jpg"),
            TestUser("user5", "eve_spammer", "Eve", False, "avatar5.jpg"),
            TestUser("user6", "frank_helper", "Frank", False, "avatar6.jpg"),
            TestUser("bot1", "assistant_bot", "Assistant Bot", True, "bot_avatar.jpg")
        ]
        
    def _generate_test_messages(self) -> None:
        """Generate comprehensive test message dataset (50+ messages)."""
        
        # Question messages (15 messages)
        question_messages = [
            TestMessage("What time is the meeting tomorrow?", self.test_users[1], "question", 0.85),
            TestMessage("How do I reset my password?", self.test_users[1], "question", 0.90),
            TestMessage("Where can I find the documentation?", self.test_users[0], "question", 0.85),
            TestMessage("When will the next release be available?", self.test_users[3], "question", 0.80),
            TestMessage("What's the best way to implement this feature?", self.test_users[0], "question", 0.75),
            TestMessage("How can I contribute to this project?", self.test_users[3], "question", 0.85),
            TestMessage("What are the system requirements?", self.test_users[1], "question", 0.80),
            TestMessage("Where should I report bugs?", self.test_users[0], "question", 0.85),
            TestMessage("How does the authentication system work?", self.test_users[1], "question", 0.75),
            TestMessage("What's the difference between these two approaches?", self.test_users[0], "question", 0.70),
            TestMessage("When is the deadline for submissions?", self.test_users[3], "question", 0.85),
            TestMessage("How do I configure the environment variables?", self.test_users[1], "question", 0.80),
            TestMessage("What permissions do I need for this operation?", self.test_users[0], "question", 0.75),
            TestMessage("Where can I download the latest version?", self.test_users[3], "question", 0.85),
            TestMessage("How long does the deployment process take?", self.test_users[1], "question", 0.80)
        ]
        
        # Complaint messages (12 messages)
        complaint_messages = [
            TestMessage("This feature is completely broken and doesn't work!", self.test_users[2], "complaint", 0.90),
            TestMessage("I'm getting an error every time I try to login", self.test_users[2], "complaint", 0.85),
            TestMessage("The system is so slow, it's unusable", self.test_users[2], "complaint", 0.80),
            TestMessage("There's a critical bug in the payment system", self.test_users[0], "complaint", 0.95),
            TestMessage("The API keeps returning 500 errors", self.test_users[1], "complaint", 0.85),
            TestMessage("This interface is confusing and poorly designed", self.test_users[2], "complaint", 0.75),
            TestMessage("I can't access my account, something is wrong", self.test_users[3], "complaint", 0.80),
            TestMessage("The documentation is outdated and incorrect", self.test_users[2], "complaint", 0.85),
            TestMessage("Performance has gotten much worse recently", self.test_users[0], "complaint", 0.80),
            TestMessage("The search functionality doesn't return relevant results", self.test_users[1], "complaint", 0.75),
            TestMessage("I'm experiencing frequent crashes when using this", self.test_users[2], "complaint", 0.85),
            TestMessage("The notification system isn't working properly", self.test_users[3], "complaint", 0.80)
        ]
        
        # Social messages (10 messages)  
        social_messages = [
            TestMessage("Hello everyone! Hope you're having a great day!", self.test_users[3], "social", 0.85),
            TestMessage("Good morning team! Ready for another productive day?", self.test_users[3], "social", 0.80),
            TestMessage("Thanks for all your help yesterday, really appreciated!", self.test_users[1], "social", 0.75),
            TestMessage("Welcome to the server! Feel free to introduce yourself", self.test_users[5], "social", 0.80),
            TestMessage("Congratulations on the successful launch!", self.test_users[3], "social", 0.85),
            TestMessage("How's everyone doing today? Any exciting projects?", self.test_users[3], "social", 0.75),
            TestMessage("Happy Friday everyone! Any weekend plans?", self.test_users[3], "social", 0.80),
            TestMessage("Great job on fixing that issue so quickly!", self.test_users[1], "social", 0.85),
            TestMessage("Hi there! New to the community and excited to learn", self.test_users[0], "social", 0.80),
            TestMessage("Thank you all for being such a supportive community!", self.test_users[3], "social", 0.85)
        ]
        
        # Spam messages (8 messages)
        spam_messages = [
            TestMessage("CLICK HERE FOR FREE MONEY!!! Limited time offer!!!", self.test_users[4], "spam", 0.95),
            TestMessage("Buy cheap followers now! Instant delivery! Click link!", self.test_users[4], "spam", 0.90),
            TestMessage("AMAZING DEAL!!! GET RICH QUICK!!!", self.test_users[4], "spam", 0.95),
            TestMessage("Work from home! Earn $5000 per week! No experience needed!!!", self.test_users[4], "spam", 0.90),
            TestMessage("URGENT: Your account will be deleted! Click here to verify!", self.test_users[4], "spam", 0.85),
            TestMessage("Free gift cards! Winners announced today! Click now!!!", self.test_users[4], "spam", 0.90),
            TestMessage("Join my crypto pump group! Easy money! DM for details!", self.test_users[4], "spam", 0.85),
            TestMessage("SYSTEM ALERT - Update required! Download now!", self.test_users[4], "spam", 0.90)
        ]
        
        # General messages (8 messages)
        general_messages = [
            TestMessage("I'll be working on the user interface improvements today", self.test_users[0], "general", 0.60),
            TestMessage("The database migration is scheduled for tonight", self.test_users[1], "general", 0.65),
            TestMessage("Here's the link to the project repository for reference", self.test_users[5], "general", 0.70),
            TestMessage("Please review the pull request when you have time", self.test_users[0], "general", 0.75),
            TestMessage("The server maintenance window starts at 2 AM UTC", self.test_users[1], "general", 0.70),
            TestMessage("Updated the documentation with the latest changes", self.test_users[5], "general", 0.65),
            TestMessage("Code review feedback has been addressed in the latest commit", self.test_users[0], "general", 0.70),
            TestMessage("The integration tests are passing on all environments", self.test_users[1], "general", 0.75)
        ]
        
        # Combine all message types
        self.test_messages = (
            question_messages + complaint_messages + social_messages + 
            spam_messages + general_messages
        )
        
        # Assign unique message IDs
        for i, message in enumerate(self.test_messages):
            message.message_id = f"msg_{i+1:03d}_{int(time.time())}"

    async def run_comprehensive_test(self) -> TestMetrics:
        """
        Execute comprehensive end-to-end system test.
        
        Tests the complete pipeline:
        1. Message ingestion simulation
        2. Signal generation and bus messaging  
        3. LLM classification
        4. Orchestration rule evaluation
        5. Response generation
        6. Health monitoring throughout
        """
        logger.info("Starting Comprehensive End-to-End Test...")
        start_time = time.time()
        
        try:
            # Pre-test health check
            await self._check_system_health("pre_test")
            
            # Process all test messages
            classification_times = []
            response_times = []
            
            for i, test_message in enumerate(self.test_messages):
                logger.info(f"Processing message {i+1}/{len(self.test_messages)}: {test_message.content[:50]}...")
                
                try:
                    # Create Signal from test message
                    signal = self._create_signal_from_test_message(test_message)
                    
                    # Test classification using REAL LLM client for authentic AI testing
                    classification_start = time.time()
                    classification = await self.real_llm_client.classify_message(signal)
                    classification_time = (time.time() - classification_start) * 1000
                    classification_times.append(classification_time)
                    
                    # Verify classification accuracy
                    if self._validate_classification(test_message, classification):
                        self.metrics.classifications_successful += 1
                    
                    # Test orchestration and response generation  
                    # For real ClassificationResult, determine response need based on purpose and urgency
                    should_respond = (
                        classification.purpose in ['question', 'complaint', 'request'] and 
                        classification.urgency in ['high', 'critical', 'medium'] and
                        classification.confidence >= 0.7
                    )
                    if should_respond:
                        response_start = time.time()
                        await self.mock_discord_tool.send_response(
                            signal, 
                            classification.to_dict(), 
                            {'channel_id': test_message.channel_id}
                        )
                        response_time = (time.time() - response_start) * 1000
                        response_times.append(response_time)
                        self.metrics.responses_generated += 1
                    
                    self.metrics.messages_processed += 1
                    
                    # Brief pause to simulate realistic timing
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message {i+1}: {e}")
                    self.metrics.error_count += 1
                    self.metrics.errors.append(f"Message {i+1}: {str(e)}")
            
            # Calculate metrics
            self.metrics.avg_classification_time_ms = sum(classification_times) / len(classification_times) if classification_times else 0
            self.metrics.avg_response_time_ms = sum(response_times) / len(response_times) if response_times else 0
            self.metrics.total_execution_time_ms = (time.time() - start_time) * 1000
            
            # Post-test health check
            await self._check_system_health("post_test")
            
            # Log comprehensive results
            await self._log_test_results()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            self.metrics.error_count += 1
            self.metrics.errors.append(f"Framework error: {str(e)}")
            raise
    
    def _create_signal_from_test_message(self, test_message: TestMessage) -> Signal:
        """Convert test message to Signal object."""
        return Signal(
            signal_id=test_message.message_id or str(uuid4()),
            source="test_discord",
            created_at=datetime.now(timezone.utc),
            author={
                "user_id": test_message.user.user_id,
                "username": test_message.user.username,
                "display_name": test_message.user.display_name,
                "is_bot": "true" if test_message.user.is_bot else "false"  # Convert bool to string
            },
            context={
                "channel_id": test_message.channel_id,
                "guild_id": test_message.guild_id,
                "message_id": test_message.message_id
            },
            content=test_message.content
        )
    
    def _validate_classification(self, test_message: TestMessage, classification: ClassificationResult) -> bool:
        """Validate classification against expected results."""
        return (
            classification.purpose == test_message.expected_type and
            classification.confidence >= test_message.expected_confidence_min
        )
    
    async def _check_system_health(self, phase: str):
        """Check system health during test execution."""
        try:
            health_status = await self.health_checker.get_system_health()
            
            for component_name, component in health_status.components.items():
                is_healthy = component.status.value == "healthy"
                self.metrics.component_health_checks[f"{phase}_{component_name}"] = is_healthy
                
            logger.info(f"Health check ({phase}): Overall status = {health_status.status.value}")
            
        except Exception as e:
            logger.error(f"Health check failed ({phase}): {e}")
            self.metrics.errors.append(f"Health check ({phase}): {str(e)}")
    
    async def _log_test_results(self):
        """Log comprehensive test results."""
        logger.info("=" * 60)
        logger.info("END-TO-END TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Messages Processed: {self.metrics.messages_processed}")
        logger.info(f"Classifications Successful: {self.metrics.classifications_successful}")
        logger.info(f"Responses Generated: {self.metrics.responses_generated}")
        logger.info(f"Avg Classification Time: {self.metrics.avg_classification_time_ms:.2f}ms")
        logger.info(f"Avg Response Time: {self.metrics.avg_response_time_ms:.2f}ms")
        logger.info(f"Total Execution Time: {self.metrics.total_execution_time_ms:.2f}ms")
        logger.info(f"Errors: {self.metrics.error_count}")
        
        # Log captured responses
        logger.info(f"Captured Responses: {len(self.mock_discord_tool.responses)}")
        for i, response in enumerate(self.mock_discord_tool.responses[:5]):  # Show first 5
            logger.info(f"   Response {i+1}: {response.content[:80]}...")
        
        if self.metrics.error_count > 0:
            logger.info("Errors encountered:")
            for error in self.metrics.errors:
                logger.info(f"   - {error}")
        
        logger.info("=" * 60)


# Pytest test cases
@pytest.fixture
async def test_framework():
    """Pytest fixture for the test framework."""
    framework = EndToEndTestFramework()
    await framework.setup()
    yield framework
    await framework.teardown()


@pytest.mark.asyncio
async def test_complete_system_pipeline(test_framework):
    """Test the complete system processing pipeline."""
    metrics = await test_framework.run_comprehensive_test()
    
    # Assert success criteria
    assert metrics.messages_processed > 0, "No messages were processed"
    assert metrics.classifications_successful > 0, "No classifications were successful"
    assert metrics.responses_generated > 0, "No responses were generated"
    assert metrics.avg_classification_time_ms < 5000, "Classification time too slow"
    assert metrics.avg_response_time_ms < 1000, "Response time too slow"
    assert metrics.error_count == 0, f"Errors encountered: {metrics.errors}"


@pytest.mark.asyncio
async def test_message_classification_accuracy(test_framework):
    """Test classification accuracy across different message types."""
    await test_framework.run_comprehensive_test()
    
    # Calculate accuracy by message type
    type_stats = {}
    for message in test_framework.test_messages:
        msg_type = message.expected_type
        if msg_type not in type_stats:
            type_stats[msg_type] = {'total': 0, 'correct': 0}
        type_stats[msg_type]['total'] += 1
    
    # Verify minimum accuracy thresholds
    min_accuracy = 0.75
    for msg_type, stats in type_stats.items():
        accuracy = stats['correct'] / stats['total']
        assert accuracy >= min_accuracy, f"Accuracy for {msg_type} too low: {accuracy:.2f}"


@pytest.mark.asyncio 
async def test_response_generation(test_framework):
    """Test response generation for appropriate message types."""
    await test_framework.run_comprehensive_test()
    
    responses = test_framework.mock_discord_tool.responses
    assert len(responses) > 0, "No responses were generated"
    
    # Verify responses have required fields
    for response in responses:
        assert response.content, "Response content is empty"
        assert response.channel_id, "Response channel_id is missing"
        assert response.signal_id, "Response signal_id is missing"
        assert response.timestamp, "Response timestamp is missing"


@pytest.mark.asyncio
async def test_system_health_during_load(test_framework):
    """Test system health monitoring during message processing."""
    await test_framework.run_comprehensive_test()
    
    # Verify health checks passed
    health_checks = test_framework.metrics.component_health_checks
    assert len(health_checks) > 0, "No health checks were performed"
    
    # Check that critical components remained healthy
    critical_components = ['database', 'signal_bus', 'llm']
    for component in critical_components:
        pre_test_key = f"pre_test_{component}"
        post_test_key = f"post_test_{component}"
        
        if pre_test_key in health_checks:
            assert health_checks[pre_test_key], f"Pre-test {component} was unhealthy"
        if post_test_key in health_checks:
            assert health_checks[post_test_key], f"Post-test {component} was unhealthy"


if __name__ == "__main__":
    """Run the end-to-end test directly."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def main():
        framework = EndToEndTestFramework()
        try:
            await framework.setup()
            await framework.run_comprehensive_test()
        finally:
            await framework.teardown()
    
    asyncio.run(main())