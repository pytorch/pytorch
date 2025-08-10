#!/usr/bin/env python3
"""
CI-compatible Python test runner for PyTorch.

This script serves as a drop-in replacement for test.sh in CI environments.
It reads the same environment variables and provides the same interface,
but uses the modern Python-based test infrastructure.

Environment Variables:
    BUILD_ENVIRONMENT: Top-level label for what's being built/tested
    TEST_CONFIG: Specific test configuration to run
    SHARD_NUMBER: Current shard number (1-based)
    NUM_TEST_SHARDS: Total number of shards
    CONTINUE_THROUGH_ERROR: Whether to continue on test failures
    VERBOSE_TEST_LOGS: Enable verbose logging
    NO_TEST_TIMEOUT: Disable test timeouts
    PYTORCH_TEST_CUDA_MEM_LEAK_CHECK: Enable CUDA memory leak checking
    PYTORCH_TEST_RERUN_DISABLED_TESTS: Rerun disabled tests
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the .ci/pytorch directory to Python path for imports
ci_dir = Path(__file__).parent
sys.path.insert(0, str(ci_dir))
# Also add parent directory to help with imports
sys.path.insert(0, str(ci_dir.parent))

try:
    # Use direct execution of simple_test_runner.py which has working imports
    import subprocess
    import json
    
    # For now, delegate to the working simple_test_runner.py
    def run_python_tests_via_simple_runner(dry_run=False, verbose=False):
        """Run tests by delegating to simple_test_runner.py which has working imports."""
        cmd = [sys.executable, str(ci_dir / "simple_test_runner.py")]
        if dry_run:
            cmd.append("--dry-run")
        if verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=ci_dir)
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Failed to run simple_test_runner.py: {e}")
            return False
    
    # Set flag to use simple runner delegation
    USE_SIMPLE_RUNNER_DELEGATION = True
    
except ImportError as e:
    print(f"Failed to import Python test infrastructure: {e}")
    print("Falling back to shell-based test.sh")
    # Fallback to original shell script
    shell_script = ci_dir / "test.sh"
    if shell_script.exists():
        os.execv("/bin/bash", ["bash", str(shell_script)] + sys.argv[1:])
    else:
        print(f"ERROR: Neither Python test runner nor shell script found")
        sys.exit(1)


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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


def check_environment() -> EnvironmentConfig:
    """Check and validate the CI environment."""
    try:
        env_config = EnvironmentConfig()
        logging.info(f"Build Environment: {env_config.build_environment}")
        logging.info(f"Test Config: {env_config.test_config}")
        logging.info(f"Shard: {env_config.shard_number}/{env_config.num_test_shards}")
        return env_config
    except Exception as e:
        logging.error(f"Failed to initialize environment config: {e}")
        raise


def run_python_tests(dry_run: bool = False, verbose: bool = False) -> bool:
    """Run tests using the Python test infrastructure."""
    try:
        # Use delegation to simple_test_runner.py which has working imports
        if 'USE_SIMPLE_RUNNER_DELEGATION' in globals() and USE_SIMPLE_RUNNER_DELEGATION:
            return run_python_tests_via_simple_runner(dry_run=dry_run, verbose=verbose)
        
        # Fallback approach (this code path may have import issues)
        from test_config.environment import EnvironmentConfig
        from test_config.test_registry import TestRegistry
        
        env_config = EnvironmentConfig()
        registry = TestRegistry()
        selected_suites = registry.select_test_suites(env_config)
        
        if not selected_suites:
            logging.warning("No test suites selected for current environment")
            return True
        
        logging.info(f"Selected {len(selected_suites)} test suite(s):")
        for suite in selected_suites:
            logging.info(f"  - {suite.name}: {suite.description}")
        
        # Execute selected test suites
        overall_success = True
        continue_on_error = os.environ.get('CONTINUE_THROUGH_ERROR', '').lower() in ('1', 'true')
        
        for suite in selected_suites:
            logging.info(f"Running test suite: {suite.name}")
            
            if dry_run:
                logging.info(f"DRY RUN - Would execute test suite: {suite.name}")
                test_names = suite.get_test_names()
                for test_name in test_names:
                    logging.info(f"  - {test_name}")
                continue
            
            try:
                success = suite.run(env_config, dry_run=dry_run)
                if not success:
                    overall_success = False
                    logging.error(f"Test suite {suite.name} failed")
                    if not continue_on_error:
                        logging.error("Stopping execution due to test failure")
                        break
                else:
                    logging.info(f"Test suite {suite.name} completed successfully")
            except Exception as e:
                logging.error(f"Exception in test suite {suite.name}: {e}")
                overall_success = False
                if not continue_on_error:
                    logging.error("Stopping execution due to exception")
                    break
        
        return overall_success
        
    except Exception as e:
        logging.error(f"Failed to run Python tests: {e}")
        return False


def fallback_to_shell(args: list) -> int:
    """Fallback to the original shell-based test.sh script."""
    logging.warning("Falling back to shell-based test.sh")
    shell_script = Path(__file__).parent / "test.sh"
    
    if not shell_script.exists():
        logging.error(f"Shell script not found: {shell_script}")
        return 1
    
    # Execute the shell script with the same arguments
    try:
        result = run_command(["bash", str(shell_script)] + args)
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
    
    try:
        # Check environment configuration (for logging purposes)
        try:
            check_environment()
        except:
            logging.info("Environment check had issues, continuing with delegation approach")
        
        # Run Python-based tests using delegation approach
        success = run_python_tests(dry_run=args.dry_run, verbose=args.verbose)
        
        if success:
            logging.info("All tests completed successfully")
            return 0
        else:
            logging.error("Some tests failed")
            if args.fallback_on_error:
                logging.info("Attempting fallback to shell script")
                return fallback_to_shell(sys.argv[1:])
            return 1
            
    except Exception as e:
        logging.error(f"Python test runner failed: {e}")
        
        if args.fallback_on_error:
            logging.info("Attempting fallback to shell script due to error")
            return fallback_to_shell(sys.argv[1:])
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
