#!/usr/bin/env python3
"""
Real AI Testing Script

This script runs the end-to-end test framework using REAL OpenAI LLM classification
while keeping Discord mocked. This tests the actual AI logic for message classification
and orchestrator responses.

Usage:
1. Set your OpenAI API key:
   $env:OPENAI_API_KEY="your-actual-openai-api-key-here"

2. Run the test:
   python run_real_ai_test.py

Features:
- Tests real OpenAI GPT-4o-mini classification on 53+ diverse messages
- Validates classification accuracy across different message types  
- Tests orchestrator rule evaluation with real AI results
- Measures actual API response times and performance
- Captures and analyzes real AI reasoning and confidence scores
- Mock Discord responses (no actual Discord API calls)
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the test framework
from tests.test_e2e_system import EndToEndTestFramework


def check_api_key():
    """Check if OpenAI API key is configured."""
    # Check both OPENAI_API_KEY and LLM_API_KEY for flexibility
    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY')
    
    if not api_key:
        print("❌ ERROR: OpenAI API key not found!")
        print()
        print("To run real AI testing, you need to set your OpenAI API key:")
        print("PowerShell: $env:OPENAI_API_KEY=\"your-api-key-here\"")
        print("Command Prompt: set OPENAI_API_KEY=your-api-key-here")
        print("Bash: export OPENAI_API_KEY=\"your-api-key-here\"")
        print()
        print("Or set LLM_API_KEY in your .env file:")
        print("LLM_API_KEY=your-api-key-here")
        print()
        print("Get your API key from: https://platform.openai.com/account/api-keys")
        print()
        return False
    
    if api_key.startswith('sk-'):
        # Determine which variable was used
        source = "OPENAI_API_KEY" if os.environ.get('OPENAI_API_KEY') else "LLM_API_KEY (.env file)"
        print(f"✅ OpenAI API key configured via {source}: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("⚠️  WARNING: API key format doesn't look correct (should start with 'sk-')")
        return False


async def run_real_ai_test():
    """Run the end-to-end test with real AI classification."""
    
    print("🧠 REAL AI TESTING FRAMEWORK")
    print("=" * 50)
    print("Testing Discord Social Credit Bot with REAL OpenAI LLM")
    print("📋 Features:")
    print("  • Real GPT-4o-mini classification on 53+ messages")  
    print("  • Mock Discord API (no external Discord calls)")
    print("  • Actual AI reasoning and confidence analysis")
    print("  • Performance metrics with real API latency")
    print("  • Classification accuracy validation")
    print("=" * 50)
    print()
    
    # Check API key
    if not check_api_key():
        return False
        
    print()
    print("🚀 Starting Real AI Test Framework...")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"real_ai_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    framework = EndToEndTestFramework()
    try:
        await framework.setup()
        
        print("📊 Test Configuration:")
        print(f"  • Test Messages: {len(framework.test_messages)}")
        print(f"  • Test Users: {len(framework.test_users)}")
        print(f"  • Expected Classifications: questions, complaints, social, spam, general")
        print()
        
        # Run the comprehensive test
        metrics = await framework.run_comprehensive_test()
        
        # Analysis and results
        print()
        print("🎯 REAL AI TEST RESULTS ANALYSIS")
        print("=" * 50)
        
        if metrics.messages_processed > 0:
            success_rate = (metrics.classifications_successful / metrics.messages_processed) * 100
            print(f"✅ Classification Success Rate: {success_rate:.1f}%")
            print(f"📈 Avg Response Time: {metrics.avg_classification_time_ms:.0f}ms")
            print(f"💬 Responses Generated: {metrics.responses_generated}")
            
            if len(framework.mock_discord_tool.responses) > 0:
                print(f"📝 Sample AI Responses:")
                for i, response in enumerate(framework.mock_discord_tool.responses[:3]):
                    print(f"   {i+1}. {response.content[:100]}...")
        
        if metrics.error_count > 0:
            print(f"❌ Errors Encountered: {metrics.error_count}")
            
        return metrics.error_count == 0
        
    except Exception as e:
        print(f"❌ Test framework error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await framework.teardown()


if __name__ == "__main__":
    print("Discord Social Credit Bot - Real AI Testing")
    print()
    
    success = asyncio.run(run_real_ai_test())
    
    print()
    if success:
        print("SUCCESS: Real AI testing completed successfully!")
        print("RESULT: Your Discord bot's AI classification system is working!")
    else:
        print("ERROR: Real AI testing encountered issues.")
        print("TIP: Check the logs above or set your OPENAI_API_KEY environment variable.")
    
    print()
    print("Next Steps:")
    print("  - Review classification accuracy in the logs")
    print("  - Test with your own Discord server using real bot token")
    print("  - Fine-tune orchestrator rules based on AI performance")
    print("  - Monitor API usage and costs in OpenAI dashboard")