#!/usr/bin/env python3
"""
Test Runner for Discord Social Credit Bot

This script provides an easy interface to run different types of tests
including unit tests, integration tests, and end-to-end system tests.

Usage:
    python run_tests.py                    # Run all tests  
    python run_tests.py --e2e              # Run only end-to-end tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --benchmark        # Run with performance benchmarks
    python run_tests.py --stress           # Run stress/load testing
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"🚀 {description}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr.strip():
                print("STDERR:", result.stderr)
            if result.stdout.strip():
                print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {description}: {e}")
        return False


def install_test_dependencies() -> bool:
    """Install test dependencies if needed."""
    logger.info("📦 Checking test dependencies...")
    
    try:
        import pytest
        import pytest_asyncio
        logger.info("✅ Test dependencies already installed")
        return True
    except ImportError:
        logger.info("📥 Installing test dependencies...")
        return run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"],
            "Installing test dependencies"
        )


def run_unit_tests(coverage: bool = False) -> bool:
    """Run unit tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "not e2e"]
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Unit and Integration Tests")


def run_e2e_tests(verbose: bool = True) -> bool:
    """Run end-to-end system tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_e2e_system.py"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, "End-to-End System Tests")


def run_all_tests(coverage: bool = False) -> bool:
    """Run all tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "All Tests")


def run_benchmarks() -> bool:
    """Run performance benchmarks."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "--benchmark-only", "--benchmark-sort=mean"]
    return run_command(cmd, "Performance Benchmarks")


def run_stress_tests() -> bool:
    """Run stress/load testing."""
    logger.info("🔥 Running stress tests...")
    
    # For now, run E2E test with larger dataset
    cmd = [
        sys.executable, "-c", 
        """
import asyncio
import sys
sys.path.append('.')
from tests.test_e2e_system import EndToEndTestFramework

async def stress_test():
    framework = EndToEndTestFramework()
    try:
        await framework.setup()
        # Run test multiple times to simulate load
        for i in range(3):
            print(f'Stress test iteration {i+1}/3')
            await framework.run_comprehensive_test()
    finally:
        await framework.teardown()

asyncio.run(stress_test())
        """
    ]
    
    return run_command(cmd, "Stress Testing")


def run_direct_e2e() -> bool:
    """Run the E2E test directly without pytest."""
    cmd = [sys.executable, "tests/test_e2e_system.py"]
    return run_command(cmd, "Direct End-to-End Test")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for Discord Social Credit Bot")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--stress", action="store_true", help="Run stress/load tests")
    parser.add_argument("--direct", action="store_true", help="Run E2E test directly (no pytest)")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    logger.info("🧪 Discord Social Credit Bot Test Runner")
    logger.info("=" * 50)
    
    # Install dependencies unless explicitly skipped
    if not args.no_install and not install_test_dependencies():
        logger.error("❌ Failed to install test dependencies")
        return 1
    
    success = True
    
    try:
        if args.install_deps:
            # Just install dependencies
            pass
        elif args.unit:
            success = run_unit_tests(args.coverage)
        elif args.e2e:
            success = run_e2e_tests()
        elif args.direct:
            success = run_direct_e2e()
        elif args.benchmark:
            success = run_benchmarks()
        elif args.stress:
            success = run_stress_tests()
        else:
            # Run all tests by default
            success = run_all_tests(args.coverage)
        
        if success:
            logger.info("🎉 All tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())