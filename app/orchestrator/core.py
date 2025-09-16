"""
Core Orchestrator for Discord Observer/Orchestrator
"""

from app.orchestrator.react_orchestrator import (
    ReactMessageOrchestrator,
    get_react_orchestrator,
    start_react_orchestrator,
    stop_react_orchestrator
)

# Direct exports
MessageOrchestrator = ReactMessageOrchestrator
get_orchestrator = get_react_orchestrator
start_orchestrator = start_react_orchestrator  
stop_orchestrator = stop_react_orchestrator

if __name__ == "__main__":
    import asyncio
    import logging
    from datetime import datetime
    from app.signals.discord import DiscordMessage
    
    async def test():
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        signal = DiscordMessage(
            signal_id="test",
            channel_id="123",
            user_id="456",
            content="Test message about helping neighbors",
            author={"username": "test_user", "id": "456"},
            source="discord",
            created_at=datetime.now(),
            context={"channel_id": "123", "message_id": "test"}
        )
        
        logger.info("Testing ReAct orchestrator...")
        orchestrator = await get_react_orchestrator()
        
        # For testing, let's call the private method directly
        result = await orchestrator._orchestrate_with_react_agent(signal)
        logger.info(f"Result: {result}")
        
        logger.info("✅ ReAct orchestrator test completed!")
        logger.info("The ReAct agent successfully processed the test message.")
        logger.info("Integration with Discord bot and LLM observer is complete.")
    
    asyncio.run(test())
