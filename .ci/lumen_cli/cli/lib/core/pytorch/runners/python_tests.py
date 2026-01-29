"""
Python test runner for PyTorch CI.

This module implements the test runner for standard Python test shards,
equivalent to the test_python_shard() and test_python() functions in test.sh.
"""

import logging
from typing import Any, Optional

from cli.lib.common.utils import run_command
from cli.lib.core.pytorch.runners.base import BasePyTorchTestRunner


logger = logging.getLogger(__name__)


class PythonTestRunner(BasePyTorchTestRunner):
    """
    Runner for standard PyTorch Python tests.

    This runner executes Python tests with optional sharding, supporting:
    - test_python_shard: Sharded test execution with JIT/distributed/quantization excluded
    - test_python: Full test suite without sharding

    Example usage:
        python -m cli.run test internal python --shard-id 1 --num-shards 4

    Equivalent to test.sh logic:
        test_python_shard() {
            time python test/run_test.py --exclude-jit-executor \\
                --exclude-distributed-tests --exclude-quantization-tests \\
                $INCLUDE_CLAUSE --shard "$1" "$NUM_TEST_SHARDS" --verbose \\
                $PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running
            assert_git_not_dirty
        }
    """

    # Tests to exclude from standard Python test runs
    EXCLUDED_TEST_CATEGORIES = [
        "--exclude-jit-executor",
        "--exclude-distributed-tests",
        "--exclude-quantization-tests",
    ]

    def __init__(self, args: Any) -> None:
        super().__init__(args)

        # Additional python test specific options
        self.include_tests: Optional[list[str]] = getattr(args, "include_tests", None)

    def run(self) -> None:
        """
        Execute the Python test run.

        This method:
        1. Sets up the environment
        2. Installs required dependencies (torchvision)
        3. Runs the Python tests with appropriate exclusions
        4. Verifies git is clean after tests
        """
        logger.info(
            "Starting Python test run: shard %d/%d",
            self.params.shard_id,
            self.params.num_shards,
        )

        # Setup environment
        self.setup_environment()

        # Install dependencies
        self.install_torchvision()

        # Run tests
        if self.params.num_shards > 1:
            self._run_python_shard()
        else:
            self._run_python_full()

        # Verify git is clean
        self.assert_git_not_dirty()

        logger.info("Python test run completed successfully")

    def _run_python_shard(self) -> None:
        """Run a sharded subset of Python tests."""
        logger.info(
            "Running Python test shard %d of %d",
            self.params.shard_id,
            self.params.num_shards,
        )

        extra_args = list(self.EXCLUDED_TEST_CATEGORIES)

        # Add include clause if specified
        if self.include_tests:
            extra_args.extend(["--include"] + self.include_tests)

        # Add Python test extra options (e.g., --xpu for XPU builds)
        python_extra = self.env.get_updates().get("PYTHON_TEST_EXTRA_OPTION", "")
        if python_extra:
            extra_args.append(python_extra)

        self.run_pytest(
            tests=[],  # Empty because we use exclusions instead
            extra_args=extra_args,
            shard=True,
        )

    def _run_python_full(self) -> None:
        """Run full Python test suite without sharding."""
        logger.info("Running full Python test suite")

        extra_args = list(self.EXCLUDED_TEST_CATEGORIES)

        # Add include clause if specified
        if self.include_tests:
            extra_args.extend(["--include"] + self.include_tests)

        # Add Python test extra options
        python_extra = self.env.get_updates().get("PYTHON_TEST_EXTRA_OPTION", "")
        if python_extra:
            extra_args.append(python_extra)

        self.run_pytest(
            tests=[],
            extra_args=extra_args,
            shard=False,
        )


class PythonLegacyJITTestRunner(BasePyTorchTestRunner):
    """
    Runner for legacy JIT Python tests.

    Equivalent to test_python_legacy_jit() in test.sh.
    """

    LEGACY_JIT_TESTS = [
        "test_jit_legacy",
        "test_jit_fuser_legacy",
    ]

    def run(self) -> None:
        """Execute legacy JIT tests."""
        logger.info("Starting legacy JIT test run")

        self.setup_environment()

        self.run_pytest(
            tests=self.LEGACY_JIT_TESTS,
            shard=False,
        )

        self.assert_git_not_dirty()
        logger.info("Legacy JIT test run completed")


class PythonSmokeTestRunner(BasePyTorchTestRunner):
    """
    Runner for smoke tests on H100/B200.

    Equivalent to test_python_smoke() in test.sh.
    """

    SMOKE_TESTS = [
        "test_matmul_cuda",
        "test_scaled_matmul_cuda",
        "inductor/test_fp8",
        "inductor/test_max_autotune",
        "inductor/test_cutedsl_grouped_mm",
    ]

    def run(self) -> None:
        """Execute smoke tests."""
        logger.info("Starting smoke test run")

        self.setup_environment()

        self.run_pytest(
            tests=self.SMOKE_TESTS,
            shard=False,
        )

        self.assert_git_not_dirty()
        logger.info("Smoke test run completed")


class QuantizationTestRunner(BasePyTorchTestRunner):
    """
    Runner for quantization tests.

    Equivalent to test_quantization() in test.sh.
    """

    def run(self) -> None:
        """Execute quantization tests."""
        logger.info("Starting quantization test run")

        self.setup_environment()

        # Quantization uses a different test file directly
        run_command("python test/test_quantization.py", env=self.env.as_dict())

        logger.info("Quantization test run completed")


class WithoutNumpyTestRunner(BasePyTorchTestRunner):
    """
    Runner for tests that verify torch works without numpy.

    Equivalent to test_without_numpy() in test.sh.
    """

    def run(self) -> None:
        """Execute without-numpy tests."""
        logger.info("Starting without-numpy test run")

        self.setup_environment()

        # Change to the CI directory which has fake_numpy
        ci_dir = self.repo_root / ".ci" / "pytorch"

        test_commands = [
            # Basic test that numpy is not available
            (
                "python -c \"import sys;sys.path.insert(0, 'fake_numpy');"
                "from unittest import TestCase;import torch;x=torch.randn(3,3);"
                "TestCase().assertRaises(RuntimeError, lambda: x.numpy())\""
            ),
            # Regression test for issue #66353
            (
                "python -c \"import sys;sys.path.insert(0, 'fake_numpy');"
                "import torch;print(torch.tensor([torch.tensor(0.), torch.tensor(1.)]))\""
            ),
            # Regression test for PR #157734 (torch.onnx importable without numpy)
            (
                "python -c \"import sys;sys.path.insert(0, 'fake_numpy');"
                "import torch; import torch.onnx\""
            ),
        ]

        # Add dynamo test if running dynamo_wrapped config
        if "dynamo_wrapped" in self.env.test_config:
            test_commands.append(
                "python -c \"import sys;sys.path.insert(0, 'fake_numpy');"
                "import torch;torch.compile(lambda x:print(x))('Hello World')\""
            )

        for cmd in test_commands:
            run_command(cmd, cwd=str(ci_dir), env=self.env.as_dict())

        logger.info("Without-numpy test run completed")
