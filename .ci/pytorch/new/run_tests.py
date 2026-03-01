#!/usr/bin/env python3
"""
Python test runner for PyTorch CI tests.

This script runs PyTorch tests directly in Python instead of calling shell scripts.
"""

import os
import sys
import argparse
from pathlib import Path

# Add test directory to path
TEST_DIR = Path(__file__).parent.parent.parent / "test"
sys.path.insert(0, str(TEST_DIR))


def run_python_tests(include_tests=None, shard_number=1, num_shards=1, verbose=False, dry_run=False):
    """
    Run Python tests using the run_test.py module.

    Args:
        include_tests: List of tests to include (e.g., ["test_torch", "test_nn"])
        shard_number: Shard number for parallel testing
        num_shards: Total number of shards
        verbose: Enable verbose output
        dry_run: If True, print what would be run without executing
    """
    if dry_run:
        print("# Would run Python tests with:")
        print(f"#   Tests: {include_tests or 'all tests'}")
        print(f"#   Shard: {shard_number}/{num_shards}")
        print(f"#   Verbose: {verbose}")
        print()

        # Build command that would be run
        cmd = ["python", "test/run_test.py"]
        if include_tests:
            cmd.extend(["--include"] + include_tests)
        cmd.extend(["--shard", str(shard_number), str(num_shards)])
        if verbose:
            cmd.append("--verbose")

        print(" ".join(cmd))
        return 0

    # Import the test runner
    try:
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(TEST_DIR)

        # Import run_test module
        import run_test

        # Build arguments for run_test
        args = []

        if include_tests:
            args.extend(["--include"] + include_tests)

        args.extend(["--shard", str(shard_number), str(num_shards)])

        if verbose:
            args.append("--verbose")

        # Parse arguments
        sys.argv = ["run_test.py"] + args

        # Run tests
        return run_test.main()

    finally:
        os.chdir(original_dir)


def run_smoke_tests(dry_run=False):
    """Run smoke tests for H100/B200"""
    smoke_tests = [
        "test_matmul_cuda",
        "test_scaled_matmul_cuda",
        "inductor/test_fp8",
        "inductor/test_max_autotune",
        "inductor/test_cutedsl_grouped_mm",
    ]

    print("Running smoke tests...")
    return run_python_tests(
        include_tests=smoke_tests,
        verbose=True,
        dry_run=dry_run
    )


def run_distributed_tests(shard_number=1, num_shards=1, dry_run=False):
    """Run distributed tests"""
    print(f"Running distributed tests (shard {shard_number}/{num_shards})...")

    cmd = [
        "run_test.py",
        "--distributed-tests",
        "--shard", str(shard_number), str(num_shards),
        "--verbose"
    ]

    if dry_run:
        print(" ".join(cmd))
        return 0

    os.chdir(TEST_DIR)

    # Import and run distributed tests
    import run_test
    sys.argv = cmd
    return run_test.main()


def run_inductor_tests(shard_number=1, num_shards=1, dry_run=False):
    """Run inductor tests"""
    inductor_tests = [
        "inductor/test_torchinductor",
        "inductor/test_torchinductor_opinfo",
        "inductor/test_aot_inductor",
    ]

    print(f"Running inductor tests (shard {shard_number}/{num_shards})...")
    return run_python_tests(
        include_tests=inductor_tests,
        shard_number=shard_number,
        num_shards=num_shards,
        verbose=True,
        dry_run=dry_run
    )


def run_test_config(test_config, build_environment, shard_number=1, num_shards=1, dry_run=False):
    """
    Run tests based on TEST_CONFIG.

    Args:
        test_config: Test configuration (e.g., "smoke", "distributed", "default")
        build_environment: Build environment string
        shard_number: Shard number
        num_shards: Total number of shards
        dry_run: If True, print what would be run without executing
    """
    print(f"Test config: {test_config}")
    print(f"Build environment: {build_environment}")
    print(f"Shard: {shard_number}/{num_shards}")
    print()

    # Route to appropriate test function based on config
    if test_config == "smoke":
        return run_smoke_tests(dry_run=dry_run)

    elif test_config == "distributed":
        return run_distributed_tests(shard_number, num_shards, dry_run=dry_run)

    elif test_config == "inductor" or "inductor" in test_config:
        return run_inductor_tests(shard_number, num_shards, dry_run=dry_run)

    elif test_config == "default":
        # Run standard test suite
        print("Running default test suite...")
        return run_python_tests(
            shard_number=shard_number,
            num_shards=num_shards,
            verbose=True,
            dry_run=dry_run
        )

    else:
        print(f"Unknown test config: {test_config}")
        print("Running default test suite...")
        return run_python_tests(
            shard_number=shard_number,
            num_shards=num_shards,
            verbose=True,
            dry_run=dry_run
        )


def main():
    parser = argparse.ArgumentParser(
        description="Python test runner for PyTorch CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run smoke tests
  BUILD_ENVIRONMENT=linux-jammy-cuda12.8-py3.10-gcc11-sm90 \\
  TEST_CONFIG=smoke \\
  python run_tests.py

  # Run distributed tests with sharding
  BUILD_ENVIRONMENT=linux-jammy-cuda12.8-py3.10-gcc11-sm90 \\
  TEST_CONFIG=distributed \\
  SHARD_NUMBER=1 \\
  NUM_TEST_SHARDS=2 \\
  python run_tests.py

  # Run specific tests
  python run_tests.py --include test_torch test_nn --verbose
        """
    )

    parser.add_argument(
        "--include",
        nargs="+",
        help="Specific tests to include"
    )

    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard number (overrides SHARD_NUMBER env var)"
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards (overrides NUM_TEST_SHARDS env var)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )

    args = parser.parse_args()

    # Get configuration from environment
    build_environment = os.environ.get("BUILD_ENVIRONMENT", "")
    test_config = os.environ.get("TEST_CONFIG", "default")

    # Get shard configuration
    shard_number = args.shard or int(os.environ.get("SHARD_NUMBER", "1"))
    num_shards = args.num_shards or int(os.environ.get("NUM_TEST_SHARDS", "1"))

    if not build_environment:
        print("Warning: BUILD_ENVIRONMENT not set, using empty string")

    # Run specific tests if requested
    if args.include:
        return run_python_tests(
            include_tests=args.include,
            shard_number=shard_number,
            num_shards=num_shards,
            verbose=args.verbose,
            dry_run=args.dry_run
        )

    # Otherwise run based on TEST_CONFIG
    return run_test_config(test_config, build_environment, shard_number, num_shards, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
