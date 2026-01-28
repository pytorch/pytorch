"""
Environment configuration for PyTorch CI tests.

This module provides centralized environment variable management for PyTorch tests,
replacing the scattered environment setup in test.sh with a structured, testable approach.

This module focuses ONLY on environment variable configuration.
For test-specific setup (CPU affinity, inductor config, etc.), see test_setup.py.

Usage:
    from cli.lib.core.pytorch import PytorchTestEnvironment

    # Option 1: Read from environment variables (CI mode)
    env = PytorchTestEnvironment()

    # Option 2: Pass values directly (CLI mode, overrides env vars)
    env = PytorchTestEnvironment(
        build_environment="linux-focal-cuda12.1-py3.10",
        test_config="inductor",
        shard_number=1,
        num_test_shards=4,
    )

    env.apply()  # Apply to current process
    env.verify_build_configuration()  # Verify build environment

    # Or pass to subprocess
    run_command("python test/run_test.py ...", env=env.as_dict())
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .properties import (
    BuildEnvironmentProperties,
    DerivedProperties,
    TestConfigProperties,
)
from .env_setup import EnvironmentSetupMixin
from .env_verification import BuildVerificationMixin


logger = logging.getLogger(__name__)


def _get_env_str(key: str, default: str = "") -> str:
    """Get environment variable as string."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class PytorchTestEnvironment(
    BuildEnvironmentProperties,
    TestConfigProperties,
    DerivedProperties,
    EnvironmentSetupMixin,
    BuildVerificationMixin,
):
    """
    Centralized environment configuration for PyTorch tests.

    This class reads CI environment variables and computes derived settings
    based on BUILD_ENVIRONMENT and TEST_CONFIG patterns.

    Values can be provided directly (CLI mode) or read from environment
    variables (CI mode). Direct values take precedence over env vars.

    Attributes:
        build_environment: The CI build environment string (e.g., "linux-focal-cuda12.1-py3.10")
        test_config: The test configuration name (e.g., "default", "slow", "inductor")
        shard_number: Current test shard number (1-indexed)
        num_test_shards: Total number of test shards

    Example:
        # CI mode - reads from environment
        env = PytorchTestEnvironment()

        # CLI mode - explicit values override environment
        env = PytorchTestEnvironment(
            build_environment="linux-focal-cuda12.1-py3.10",
            test_config="inductor",
            shard_number=1,
            num_test_shards=4,
        )

        env.apply()  # Apply to current process
    """

    # These can be passed directly or read from environment variables.
    # Direct values (not None) take precedence over env vars.
    build_environment: Optional[str] = None
    test_config: Optional[str] = None
    shard_number: Optional[int] = None
    num_test_shards: Optional[int] = None

    # Internal state
    _env_updates: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """
        Compute all derived environment variables after initialization.

        If values are not provided directly, they are read from environment variables.
        """
        # Resolve values: direct input takes precedence over env vars
        if self.build_environment is None:
            self.build_environment = _get_env_str("BUILD_ENVIRONMENT", "")
        if self.test_config is None:
            self.test_config = _get_env_str("TEST_CONFIG", "default")
        if self.shard_number is None:
            self.shard_number = _get_env_int("SHARD_NUMBER", 1)
        if self.num_test_shards is None:
            self.num_test_shards = _get_env_int("NUM_TEST_SHARDS", 1)

        # Initialize env updates dict
        self._env_updates = {}

        # Run all setup methods from EnvironmentSetupMixin
        self.run_all_setup()

        logger.debug(
            "TestEnvironment initialized: build_environment=%s, test_config=%s, "
            "shard=%d/%d",
            self.build_environment,
            self.test_config,
            self.shard_number,
            self.num_test_shards,
        )
        logger.debug("Computed env updates: %s", self._env_updates)

    @classmethod
    def from_args(cls, args) -> "PytorchTestEnvironment":
        """
        Create TestEnvironment from parsed CLI arguments.

        This factory method extracts relevant arguments from an argparse namespace
        and creates a TestEnvironment instance.

        Args:
            args: Parsed CLI arguments (argparse.Namespace) with optional attributes:
                - build_environment: str
                - test_config: str
                - shard_id or shard_number: int
                - num_shards or num_test_shards: int

        Returns:
            TestEnvironment configured from the CLI arguments.

        Example:
            args = parser.parse_args()
            env = TestEnvironment.from_args(args)
        """
        return cls(
            build_environment=getattr(args, "build_environment", None),
            test_config=getattr(args, "test_config", None),
            shard_number=getattr(args, "shard_id", None)
            or getattr(args, "shard_number", None),
            num_test_shards=getattr(args, "num_shards", None)
            or getattr(args, "num_test_shards", None),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def apply(self) -> None:
        """Apply all computed environment variables to the current process."""
        os.environ.update(self._env_updates)
        logger.info("Applied %d environment variables", len(self._env_updates))

    def as_dict(self) -> dict[str, str]:
        """
        Return a complete environment dict for passing to subprocess.

        This merges the current process environment with computed updates.
        """
        return {**os.environ, **self._env_updates}

    def get_updates(self) -> dict[str, str]:
        """Return only the computed environment variable updates."""
        return self._env_updates.copy()

    def get_categorized_updates(self) -> dict[str, dict[str, str]]:
        """
        Return environment variables grouped by category for display purposes.

        Returns:
            A dictionary with category names as keys and dicts of env vars as values.
        """
        # Define which keys belong to directory-related variables
        directory_keys = {
            "BUILD_DIR",
            "BUILD_RENAMED_DIR",
            "BUILD_BIN_DIR",
            "TORCH_INSTALL_DIR",
            "TORCH_BIN_DIR",
            "TORCH_LIB_DIR",
            "TORCH_TEST_DIR",
        }

        directory_vars = {}
        other_vars = {}

        for key, value in self._env_updates.items():
            if key in directory_keys:
                directory_vars[key] = value
            else:
                other_vars[key] = value

        return {
            "Directory Environment Variables": directory_vars,
            "Compute Environment Variables": other_vars,
        }

    def set(self, key: str, value: str) -> None:
        """Set an additional environment variable."""
        self._env_updates[key] = value

    def unset(self, key: str) -> None:
        """Remove an environment variable from updates."""
        self._env_updates.pop(key, None)
