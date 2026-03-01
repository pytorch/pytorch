#!/usr/bin/env python3
"""
Complete PyTorch CI test pipeline in Python.

This script combines setup and test execution in one command.
All environment setup happens in the parent process, then tests
inherit the configured environment.
"""

import os
import sys
import argparse
import inspect
from pathlib import Path

# Add parent directory and test directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import all needed functions from mapping
from mapping import (
    setup_test_environment,
    print_as_shell_commands,
    get_torch_directories,
    get_build_directories,
    set_default_shard_config,
    SPECIAL_OPERATIONS,
)

# Import test runner
import run_tests


def main():
    parser = argparse.ArgumentParser(
        description="Complete PyTorch CI test pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: setup + run smoke tests
  BUILD_ENVIRONMENT=linux-jammy-cuda12.8-py3.10-gcc11-sm90 \\
  TEST_CONFIG=smoke \\
  python ci_pipeline.py

  # Just setup (dry-run, print what would be done)
  BUILD_ENVIRONMENT=linux-jammy-cuda12.8-py3.10-gcc11-sm90 \\
  TEST_CONFIG=smoke \\
  python ci_pipeline.py --setup-only --dry-run

  # Full pipeline with dry-run
  BUILD_ENVIRONMENT=linux-jammy-cuda12.8-py3.10-gcc11-sm90 \\
  TEST_CONFIG=smoke \\
  python ci_pipeline.py --dry-run
        """
    )

    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only run setup, don't run tests"
    )

    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="Only run tests, skip setup (assumes setup already done)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )

    args = parser.parse_args()

    # Get configuration from environment
    build_environment = os.environ.get("BUILD_ENVIRONMENT", "")
    test_config = os.environ.get("TEST_CONFIG", "default")
    shard_number = int(os.environ.get("SHARD_NUMBER", "1"))
    num_shards = int(os.environ.get("NUM_TEST_SHARDS", "1"))

    if not build_environment:
        print("ERROR: BUILD_ENVIRONMENT is required", file=sys.stderr)
        return 1

    print("=" * 80)
    print("PyTorch CI Pipeline")
    print("=" * 80)
    print(f"BUILD_ENVIRONMENT: {build_environment}")
    print(f"TEST_CONFIG: {test_config}")
    print(f"SHARD: {shard_number}/{num_shards}")
    print("=" * 80)
    print()

    # Step 1: Setup
    if not args.tests_only:
        print("STEP 1: Environment Setup")
        print("-" * 80)

        if args.dry_run:
            # Dry-run: just print what would be done
            env_vars, operations = setup_test_environment(build_environment, test_config)

            # Add shard config
            env_vars["SHARD_NUMBER"] = str(shard_number)
            env_vars["NUM_TEST_SHARDS"] = str(num_shards)

            # Add directory paths
            torch_dirs = get_torch_directories()
            build_dirs = get_build_directories()
            env_vars.update(torch_dirs)
            env_vars.update(build_dirs)

            # Print as shell commands
            print_as_shell_commands(env_vars, operations, build_environment, test_config)
        else:
            # Actually execute setup IN THIS PROCESS
            # This ensures all env vars are set before spawning test subprocess
            print("Executing setup in parent process...")
            print("(Environment variables and patches will be inherited by test subprocess)")
            print()

            # Set up shard configuration
            shard_config = set_default_shard_config()

            # Get environment configuration
            env_vars, operations = setup_test_environment(build_environment, test_config)

            # Add shard config to env vars
            env_vars["SHARD_NUMBER"] = str(shard_config["SHARD_NUMBER"])
            env_vars["NUM_TEST_SHARDS"] = str(shard_config["NUM_TEST_SHARDS"])

            # Add directory paths
            torch_dirs = get_torch_directories()
            build_dirs = get_build_directories()
            env_vars.update(torch_dirs)
            env_vars.update(build_dirs)

            # Apply environment variables to current process
            print("Applying environment variables to current process...")
            for key, value in sorted(env_vars.items()):
                os.environ[key] = value
                print(f"  Set {key}={value[:80]}...")

            print()

            # Execute special operations in current process
            print("Executing special operations in current process...")
            for op_name, op_info in operations.items():
                if not op_info["enabled"]:
                    continue

                print(f"  {op_info['description']}...")

                # Find the operation and execute its executor function
                op = next((o for o in SPECIAL_OPERATIONS if o.name == op_name), None)
                if op and op.executor:
                    try:
                        # Check if function needs build_environment parameter
                        sig = inspect.signature(op.executor)
                        if len(sig.parameters) > 0:
                            op.executor(build_environment)
                        else:
                            op.executor()
                    except Exception as e:
                        print(f"    Warning: Operation {op_name} failed: {e}")
                else:
                    print(f"    Warning: No executor found for {op_name}")

            print()
            print("Setup complete! Environment is configured for test subprocess.")

        print()

    if args.setup_only:
        print("Setup complete (--setup-only, skipping tests)")
        return 0

    # Step 2: Run tests
    # Tests will run in subprocess and inherit all environment from this process
    print("STEP 2: Running Tests")
    print("-" * 80)

    if args.dry_run:
        print("Dry-run mode: showing what tests would be run")
        print()
        print("Tests will inherit all environment variables from parent process")
        print()

        # Show what tests would run
        return run_tests.run_test_config(
            test_config,
            build_environment,
            shard_number,
            num_shards,
            dry_run=True
        )
    else:
        print("Tests will inherit all environment variables from this process")
        print()

        try:
            return run_tests.run_test_config(
                test_config,
                build_environment,
                shard_number,
                num_shards,
                dry_run=False
            )
        except Exception as e:
            print(f"\nTests failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
