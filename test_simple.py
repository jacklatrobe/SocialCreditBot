#!/usr/bin/env python3
"""
Simple Discord Bot Test - Training Data Collection

Simplified version that focuses on testing the training data collection system.
This bypasses some of the complex orchestration and focuses on the core functionality.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('simple_test.log', encoding='utf-8')]
)

logger = logging.getLogger(__name__)


async def test_discord_connection():
    """Test basic Discord connection and message processing."""
    try:
        print("Testing Discord connection...")
        
        # Import components
        from app.config import get_settings
        from app.infra.bus import init_signal_bus
        from app.ingest.discord_client import DiscordIngestClient
        
        # Get config
        config = get_settings()
        print(f"✓ Config loaded - Discord token: {'YES' if config.discord_bot_token else 'NO'}")
        print(f"✓ Training data enabled: {config.training_data_enabled}")
        print(f"✓ Training data path: {config.training_data_path}")
        
        # Initialize signal bus
        signal_bus = await init_signal_bus()
        print(f"✓ Signal bus running: {signal_bus.running}")
        
        # Initialize Discord client
        client = DiscordIngestClient()
        init_success = await client.initialize()
        print(f"✓ Discord client initialized: {init_success}")
        
        if init_success:
            # Connect to signal bus
            bus_success = await client.connect_signal_bus()
            print(f"✓ Signal bus connected: {bus_success}")
            
            if bus_success:
                print("\\n[READY] Bot is ready to connect to Discord!")
                print("Press Ctrl+C to stop the test...")
                
                # Start the Discord client (this will connect to Discord)
                start_success = await client.start()
                print(f"Discord connection result: {start_success}")
            else:
                print("✗ Failed to connect to signal bus")
        else:
            print("✗ Failed to initialize Discord client")
            
    except KeyboardInterrupt:
        print("\\n[STOPPED] Test stopped by user")
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Simple Discord Bot Test ===")
    print("This will test Discord connection and basic message processing.")
    print("")
    
    try:
        asyncio.run(test_discord_connection())
    except KeyboardInterrupt:
        print("Goodbye!")
    except Exception as e:
        print(f"Test failed: {e}")