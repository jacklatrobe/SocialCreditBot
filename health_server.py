#!/usr/bin/env python3
"""
Health Check Server Startup Script

This script starts the FastAPI-based health check server for monitoring
the Discord Social Credit Bot system components.

Usage:
    python health_server.py [--host HOST] [--port PORT] [--reload]

Examples:
    python health_server.py                    # Start on localhost:8001
    python health_server.py --port 8080       # Start on localhost:8080
    python health_server.py --host 0.0.0.0    # Start on all interfaces
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn is required to run the health check server.")
    print("Install it with: pip install uvicorn")
    sys.exit(1)

from app.monitoring.health import app
from app.config import Settings as Config


def setup_logging():
    """Configure logging for the health check server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('health_check.log')
        ]
    )


def main():
    """Main entry point for health check server"""
    parser = argparse.ArgumentParser(
        description="Discord Social Credit Bot Health Check Server"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001, 
        help="Port to bind the server to (default: 8001)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Set the logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration to validate it exists
    try:
        config = Config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    logger.info(f"Starting health check server on {args.host}:{args.port}")
    logger.info("Health check endpoints available at:")
    logger.info(f"  - Simple health check: http://{args.host}:{args.port}/health/simple")
    logger.info(f"  - Full health check: http://{args.host}:{args.port}/health")
    logger.info(f"  - Detailed status: http://{args.host}:{args.port}/health/detailed")
    logger.info(f"  - API documentation: http://{args.host}:{args.port}/docs")
    
    # Start the FastAPI server with uvicorn
    try:
        uvicorn.run(
            "app.monitoring.health:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Health check server stopped by user")
    except Exception as e:
        logger.error(f"Health check server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()