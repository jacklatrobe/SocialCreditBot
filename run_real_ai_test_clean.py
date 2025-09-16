#!/usr/bin/env python3
"""
Real AI Testing Script for Discord Social Credit Bot

This script tests the complete system using:
- REAL OpenAI LLM for authentic AI classification testing  
- MOCK Discord API to avoid external calls and costs
- 53+ diverse test messages across different categories

Usage: python run_real_ai_test_clean.py

Requires: OPENAI_API_KEY in environment or .env file
"""

import asyncio
import os
import sys
import logging
from typing import Tuple
from dotenv import load_dotenv

# Configure Python to handle Unicode properly on Windows
if sys.platform.startswith('win'):
    # Set environment to use UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load environment variables from .env file
load_dotenv()

# Configure logging to be ASCII-safe for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set logging to handle Unicode gracefully
for handler in logging.getLogger().handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'encoding'):
        if handler.stream.encoding != 'utf-8':
            handler.stream.reconfigure(encoding='utf-8', errors='replace')

def check_api_key() -> Tuple[bool, str]:
    """
    Check if OpenAI API key is properly configured
    Returns: (is_configured, message)
    """
    # Try different possible environment variable names
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    
    if not api_key:
        return False, "ERROR: OpenAI API key not found!"
    
    # Determine source
    source = ".env file"
    if os.getenv("OPENAI_API_KEY"):
        source = "OPENAI_API_KEY variable"
    elif os.getenv("LLM_API_KEY"):
        source = "LLM_API_KEY variable"
    
    if len(api_key) > 20:
        print(f"SUCCESS: OpenAI API key configured via {source}: {api_key[:10]}...{api_key[-4:]}")
        
        if not api_key.startswith('sk-'):
            print("WARNING: API key format doesn't look correct (should start with 'sk-')")
        
        return True, f"API key found in {source}"
    else:
        return False, f"API key found but appears invalid (too short: {len(api_key)} chars)"


async def run_real_ai_test():
    """Execute the real AI testing framework"""
    print("REAL AI TESTING FRAMEWORK")
    print("=" * 50)
    print()
    print("Features:")
    print("  - Real GPT-4o-mini classification on 53+ messages")  
    print("  - Mock Discord API (no external Discord calls)")
    print("  - Actual AI reasoning and confidence analysis")
    print("  - Performance metrics with real API latency")
    print("  - Classification accuracy validation")
    print()
    
    try:
        # Import here to avoid circular imports and ensure .env is loaded
        from tests.test_e2e_system import EndToEndTestFramework
        
        print("Starting Real AI Test Framework...")
        print()
        
        # Initialize the test framework
        framework = EndToEndTestFramework()
        
        await framework.setup()
        
        # Display test configuration
        print("Test Configuration:")
        print(f"  - Test Messages: {len(framework.test_messages)}")
        print(f"  - Test Users: {len(framework.test_users)}")
        print(f"  - Expected Classifications: questions, complaints, social, spam, general")
        print()
        
        # Execute the real AI tests
        results = await framework.run_comprehensive_test()
        
        # Analyze and display results
        print("=" * 50)
        print("REAL AI TEST RESULTS ANALYSIS")
        print("=" * 50)
        
        if results and hasattr(results, 'metrics'):
            metrics = results.metrics
            success_rate = (metrics.successful_classifications / len(framework.test_messages)) * 100
            print(f"SUCCESS: Classification Success Rate: {success_rate:.1f}%")
            print(f"METRICS: Avg Response Time: {metrics.avg_classification_time_ms:.0f}ms")
            print(f"RESPONSES: AI Responses Generated: {metrics.responses_generated}")
            print(f"TOTAL: Processing Time: {metrics.total_execution_time_ms/1000:.1f}s")
            
            if hasattr(results, 'sample_responses') and results.sample_responses:
                print(f"SAMPLES: Sample AI Responses:")
                for i, response in enumerate(results.sample_responses[:3], 1):
                    print(f"  {i}. {response[:80]}...")
            
            print()
            print(f"RESULTS: Processed {metrics.successful_classifications} out of {len(framework.test_messages)} messages")
            print(f"ERRORS: {metrics.failed_classifications} failed classifications")
            
            # Consider test successful if we got reasonable results
            return success_rate > 40  # At least 40% classification success
        else:
            print("ERROR: No results returned from test framework")
            return False
            
    except Exception as e:
        print(f"ERROR: Real AI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'framework' in locals():
            await framework.teardown()


if __name__ == "__main__":
    print("Discord Social Credit Bot - Real AI Testing")
    print()
    
    # Check API key first
    api_configured, api_message = check_api_key()
    if not api_configured:
        print("SETUP REQUIRED:")
        print(f"   {api_message}")
        print()
        print("Setup Instructions:")
        print("   1. Get OpenAI API key from: https://platform.openai.com/api-keys")
        print("   2. Create .env file in project root with:")
        print("      OPENAI_API_KEY=your_api_key_here")
        print("   3. Ensure .env file is in .gitignore to keep API key secure")
        print()
        exit(1)
    
    success = asyncio.run(run_real_ai_test())
    
    print()
    if success:
        print("SUCCESS: Real AI testing completed successfully!")
        print("RESULT: Your Discord bot's AI classification system is working!")
    else:
        print("ERROR: Real AI testing encountered issues.")
        print("TIP: Check the logs above or verify your OPENAI_API_KEY.")
    
    print()
    print("Next Steps:")
    print("  - Review classification accuracy in the logs")
    print("  - Test with your own Discord server using real bot token")
    print("  - Fine-tune orchestrator rules based on AI performance")
    print("  - Monitor API usage and costs in OpenAI dashboard")