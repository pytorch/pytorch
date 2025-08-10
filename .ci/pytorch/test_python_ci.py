#!/usr/bin/env python3
"""
CI-compatible Python test runner for PyTorch.

This script serves as a drop-in replacement for test.sh in CI environments.
It delegates to the working simple_test_runner.py to avoid import issues.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PyTorch CI Python Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--fallback-on-error',
        action='store_true',
        default=True,
        help='Fallback to shell script on Python test runner errors'
    )
    
    return parser.parse_args()


def run_python_tests_via_delegation(dry_run: bool = False, verbose: bool = False) -> int:
    """Run tests by delegating to simple_test_runner.py which has working imports.
    
    Returns:
        int: The exit code from the delegated test runner (0 for success, non-zero for failure)
    """
    ci_dir = Path(__file__).parent
    simple_runner = ci_dir / "simple_test_runner.py"
    
    if not simple_runner.exists():
        logging.error(f"simple_test_runner.py not found at {simple_runner}")
        return 1
    
    cmd = [sys.executable, str(simple_runner)]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")
    
    try:
        logging.info(f"Delegating to: {' '.join(cmd)}")
        # Properly capture and propagate the exit code
        result = subprocess.run(cmd, cwd=ci_dir, capture_output=False)
        logging.info(f"Delegated test runner completed with exit code: {result.returncode}")
        return result.returncode
    except Exception as e:
        logging.error(f"Failed to run simple_test_runner.py: {e}")
        return 1


def fallback_to_shell(args: list) -> int:
    """Fallback to the original shell-based test.sh script."""
    logging.warning("Falling back to shell-based test.sh")
    shell_script = Path(__file__).parent / "test.sh"
    
    if not shell_script.exists():
        logging.error(f"Shell script not found: {shell_script}")
        return 1
    
    # Execute the shell script with the same arguments
    try:
        result = subprocess.run(["bash", str(shell_script)] + args, cwd=shell_script.parent)
        return result.returncode
    except Exception as e:
        logging.error(f"Failed to execute shell script: {e}")
        return 1


def main() -> int:
    """Main entry point for the CI test runner."""
    args = parse_args()
    
    # Setup logging based on environment and arguments
    verbose = args.verbose or os.environ.get('VERBOSE_TEST_LOGS', '').lower() in ('1', 'true')
    setup_logging(verbose)
    
    logging.info("PyTorch CI Python Test Runner")
    logging.info("=" * 50)
    
    # Log environment information
    build_env = os.environ.get('BUILD_ENVIRONMENT', '')
    test_config = os.environ.get('TEST_CONFIG', '')
    shard_num = os.environ.get('SHARD_NUMBER', '1')
    num_shards = os.environ.get('NUM_TEST_SHARDS', '1')
    
    logging.info(f"Build Environment: {build_env}")
    logging.info(f"Test Config: {test_config}")
    logging.info(f"Shard: {shard_num}/{num_shards}")
    
    try:
        # Run Python-based tests via delegation
        exit_code = run_python_tests_via_delegation(dry_run=args.dry_run, verbose=verbose)
        
        if exit_code == 0:
            logging.info("All tests completed successfully")
            return 0
        else:
            logging.error(f"Tests failed with exit code: {exit_code}")
            if args.fallback_on_error:
                logging.info("Attempting fallback to shell script")
                return fallback_to_shell(sys.argv[1:])
            return exit_code
            
    except Exception as e:
        logging.error(f"Python test runner failed: {e}")
        
        if args.fallback_on_error:
            logging.info("Attempting fallback to shell script due to error")
            return fallback_to_shell(sys.argv[1:])
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
