"""
Base runner class for PyTorch internal tests.

This module provides the base class that all PyTorch test runners inherit from,
including common functionality like environment setup, git status checks, and
dependency installation.
"""

import logging
import subprocess
import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.utils import run_command
from cli.lib.core.pytorch.env_config import TestEnvironment
from cli.lib.core.pytorch.test_setup import (
    CPUAffinityConfig,
    InductorTestConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class PyTorchTestParams:
    """
    Common parameters for PyTorch test runners.

    These parameters are typically passed via CLI arguments.
    """

    shard_id: int = 1
    num_shards: int = 1
    test_config: str = ""
    include_tests: Optional[list[str]] = None
    exclude_tests: Optional[list[str]] = None
    verbose: bool = True
    upload_artifacts: bool = True


class BasePyTorchTestRunner(BaseRunner):
    """
    Base class for PyTorch internal test runners.

    This class provides common functionality shared by all test runners:
    - Environment setup via TestEnvironment
    - Git cleanliness checks
    - Dependency installation helpers
    - Common test execution patterns

    Subclasses should implement the `run()` method with their specific test logic.
    """

    def __init__(self, args: Any) -> None:
        """
        Initialize the test runner with CLI arguments.

        Args:
            args: Parsed CLI arguments containing test configuration.
        """
        super().__init__(args)

        # Extract common parameters from args
        self.params = PyTorchTestParams(
            shard_id=getattr(args, "shard_id", 1),
            num_shards=getattr(args, "num_shards", 1),
            test_config=getattr(args, "test_config", ""),
            verbose=getattr(args, "verbose", True),
            upload_artifacts=getattr(args, "upload_artifacts", True),
        )

        # Initialize environment configuration
        self.env = TestEnvironment()

        # Paths
        self.repo_root = Path.cwd()
        self.test_dir = self.repo_root / "test"
        self.test_reports_dir = self.test_dir / "test-reports"

    # =========================================================================
    # Environment Setup
    # =========================================================================

    def setup_environment(self) -> None:
        """
        Apply all environment variables for the test run.

        This method applies the TestEnvironment configuration and performs
        any additional setup required for the specific test type.
        """
        logger.info("Setting up test environment...")
        self.env.apply()

        # Ensure test reports directory exists
        self.test_reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Environment setup complete. BUILD_ENVIRONMENT=%s, TEST_CONFIG=%s",
            self.env.build_environment,
            self.env.test_config,
        )

    def setup_cpu_affinity(self) -> CPUAffinityConfig:
        """
        Setup CPU affinity for performance-sensitive tests.

        Returns:
            CPUAffinityConfig with taskset command and environment variables.
        """
        logger.info("Setting up CPU affinity...")
        cpu_config = CPUAffinityConfig()
        cpu_config.apply()
        return cpu_config

    def setup_inductor_env(self) -> InductorTestConfig:
        """
        Setup inductor-specific environment variables.

        Returns:
            InductorTestConfig with inductor-specific settings.
        """
        logger.info("Setting up inductor environment...")
        inductor_config = InductorTestConfig()
        inductor_config.apply()
        return inductor_config

    # =========================================================================
    # Dependency Installation
    # =========================================================================

    def install_torchvision(self) -> None:
        """Install torchvision from the built wheel or pip."""
        logger.info("Installing torchvision...")
        # This would call the install_torchvision function from common-build.sh
        # For now, we assume it's available or we use pip
        try:
            run_command("pip install torchvision", check=False)
        except subprocess.CalledProcessError:
            logger.warning("Failed to install torchvision, continuing anyway")

    def install_torchaudio(self) -> None:
        """Install torchaudio from the built wheel or pip."""
        logger.info("Installing torchaudio...")
        try:
            run_command("pip install torchaudio", check=False)
        except subprocess.CalledProcessError:
            logger.warning("Failed to install torchaudio, continuing anyway")

    def pip_install(self, package: str) -> None:
        """
        Install a package via pip.

        Args:
            package: Package specification (name, name==version, or path).
        """
        run_command(f"pip install {package}")

    # =========================================================================
    # Git Utilities
    # =========================================================================

    def assert_git_not_dirty(self) -> None:
        """
        Assert that the git repository is clean.

        Raises:
            RuntimeError: If there are uncommitted changes.
        """
        logger.info("Checking git status...")
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )

        if result.stdout.strip():
            dirty_files = result.stdout.strip()
            logger.error("Git repository is dirty:\n%s", dirty_files)
            raise RuntimeError(
                f"Git repository has uncommitted changes:\n{dirty_files}"
            )

        logger.debug("Git repository is clean")

    # =========================================================================
    # Test Execution Helpers
    # =========================================================================

    def run_pytest(
        self,
        tests: list[str],
        *,
        extra_args: Optional[list[str]] = None,
        shard: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> int:
        """
        Run pytest with the specified tests.

        Args:
            tests: List of test modules or paths to run.
            extra_args: Additional arguments to pass to run_test.py.
            shard: Whether to apply sharding.
            env: Additional environment variables.

        Returns:
            Exit code from the test run.
        """
        cmd_parts = [sys.executable, "test/run_test.py"]

        # Add test includes
        if tests:
            cmd_parts.extend(["--include"] + tests)

        # Add sharding
        if shard and self.params.num_shards > 1:
            cmd_parts.extend([
                "--shard",
                str(self.params.shard_id),
                str(self.params.num_shards),
            ])

        # Add verbosity
        if self.params.verbose:
            cmd_parts.append("--verbose")

        # Add artifact upload
        if self.params.upload_artifacts:
            cmd_parts.append("--upload-artifacts-while-running")

        # Add extra args
        if extra_args:
            cmd_parts.extend(extra_args)

        # Build command string
        cmd = " ".join(cmd_parts)

        # Merge environment
        run_env = self.env.as_dict()
        if env:
            run_env.update(env)

        logger.info("Running: %s", cmd)
        return run_command(cmd, env=run_env, check=False)

    def run_cpp_tests(
        self,
        tests: list[str],
        *,
        env: Optional[dict[str, str]] = None,
    ) -> int:
        """
        Run C++ tests via run_test.py.

        Args:
            tests: List of C++ test names (e.g., ["cpp/test_api", "cpp/test_jit"]).
            env: Additional environment variables.

        Returns:
            Exit code from the test run.
        """
        cmd_parts = [
            sys.executable,
            "test/run_test.py",
            "--cpp",
            "--verbose",
        ]

        if tests:
            cmd_parts.extend(["-i"] + tests)

        cmd = " ".join(cmd_parts)

        run_env = self.env.as_dict()
        if env:
            run_env.update(env)

        logger.info("Running C++ tests: %s", cmd)
        return run_command(cmd, env=run_env, check=False)

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def run(self) -> None:
        """
        Execute the test run.

        Subclasses must implement this method with their specific test logic.
        """
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_torch_paths(self) -> dict[str, Path]:
        """
        Get paths to torch installation directories.

        Returns:
            Dictionary with keys: install_dir, bin_dir, lib_dir, test_dir.
        """
        try:
            import torch

            install_dir = Path(torch.__file__).parent
            return {
                "install_dir": install_dir,
                "bin_dir": install_dir / "bin",
                "lib_dir": install_dir / "lib",
                "test_dir": install_dir / "test",
            }
        except ImportError:
            logger.warning("torch not installed, returning empty paths")
            return {
                "install_dir": Path(),
                "bin_dir": Path(),
                "lib_dir": Path(),
                "test_dir": Path(),
            }

    def print_torch_info(self) -> None:
        """Print torch configuration information."""
        try:
            import torch

            logger.info("PyTorch version: %s", torch.__version__)
            logger.info("PyTorch config:\n%s", torch.__config__.show())
            logger.info("Parallel info:\n%s", torch.__config__.parallel_info())
        except ImportError:
            logger.warning("torch not installed, skipping info")
