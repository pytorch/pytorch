#!/usr/bin/env python3
"""
PyTorch CI Test Runner

This is a Python-based replacement for the test.sh shell script.
It provides a more maintainable and extensible way to run PyTorch CI tests.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_config.environment import EnvironmentConfig
from test_config.test_registry import TestRegistry
from utils.shell_utils import run_command


def setup_logging() -> None:
    """Configure logging for the test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> int:
    """Main entry point for the test runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='PyTorch CI Test Runner')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be executed without running tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize environment configuration
        env_config = EnvironmentConfig()
        logger.info(f"Build Environment: {env_config.build_environment}")
        logger.info(f"Test Config: {env_config.test_config}")
        logger.info(f"Shard: {env_config.shard_number}/{env_config.num_test_shards}")
        
        # Get test registry and determine which tests to run
        registry = TestRegistry()
        test_suite = registry.get_test_suite(env_config)
        
        if not test_suite:
            logger.error("No test suite found for current configuration")
            return 1
        
        logger.info(f"Selected test suite: {test_suite.name}")
        
        if args.dry_run:
            logger.info("DRY RUN - Would execute the following tests:")
            for test_name in test_suite.get_test_names():
                logger.info(f"  - {test_name}")
            return 0
        
        # Execute the test suite
        success = test_suite.run(env_config)
        
        if success:
            logger.info("All tests completed successfully")
            return 0
        else:
            logger.error("Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
