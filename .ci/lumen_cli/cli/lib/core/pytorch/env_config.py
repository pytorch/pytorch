"""
Environment configuration for PyTorch CI tests.

This module provides centralized environment variable management for PyTorch tests,
replacing the scattered environment setup in test.sh with a structured, testable approach.

This module focuses ONLY on environment variable configuration.
For test-specific setup (CPU affinity, inductor config, etc.), see test_setup.py.

Usage:
    from cli.lib.core.pytorch.env_config import TestEnvironment

    # Option 1: Read from environment variables (CI mode)
    env = TestEnvironment()

    # Option 2: Pass values directly (CLI mode, overrides env vars)
    env = TestEnvironment(
        build_environment="linux-focal-cuda12.1-py3.10",
        test_config="inductor",
        shard_number=1,
        num_test_shards=4,
    )

    env.apply()  # Apply to current process

    # Or pass to subprocess
    run_command("python test/run_test.py ...", env=env.as_dict())
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional


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
class TestEnvironment:
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
        env = TestEnvironment()

        # CLI mode - explicit values override environment
        env = TestEnvironment(
            build_environment="linux-focal-cuda12.1-py3.10",
            test_config="inductor",
            shard_number=1,
            num_test_shards=4,
        )
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

        # Compute derived environment variables
        self._setup_base_env()
        self._setup_device_visibility()
        self._setup_test_flags()
        self._setup_valgrind()
        self._setup_sanitizers()
        self._setup_cpu_capability()
        self._setup_legacy_driver()

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
    def from_args(cls, args) -> "TestEnvironment":
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
    # Environment Setup Methods
    # =========================================================================

    def _setup_base_env(self) -> None:
        """Set base environment variables required for all tests."""
        # Suppress ANSI color escape sequences
        self._env_updates["TERM"] = "vt100"

        # Set locale
        self._env_updates["LANG"] = "C.UTF-8"

        # Enable debug asserts in serialization
        self._env_updates["TORCH_SERIALIZATION_DEBUG"] = "1"

    def _setup_device_visibility(self) -> None:
        """Configure GPU device visibility based on test config."""
        if self.test_config == "default":
            self._env_updates["CUDA_VISIBLE_DEVICES"] = "0"
            self._env_updates["HIP_VISIBLE_DEVICES"] = "0"

        if self.test_config == "distributed" and self.is_rocm:
            self._env_updates["HIP_VISIBLE_DEVICES"] = "0,1,2,3"

    def _setup_test_flags(self) -> None:
        """Configure PyTorch test flags based on test config and build environment."""
        # Slow tests
        if self.test_config == "slow":
            self._env_updates["PYTORCH_TEST_WITH_SLOW"] = "1"
            self._env_updates["PYTORCH_TEST_SKIP_FAST"] = "1"

        # Slow gradcheck tests
        if "slow-gradcheck" in self.build_environment:
            self._env_updates["PYTORCH_TEST_WITH_SLOW_GRADCHECK"] = "1"
            # Run tests sequentially to mitigate OOM issues
            self._env_updates["PYTORCH_TEST_CUDA_MEM_LEAK_CHECK"] = "1"

        # Device-specific test filtering
        if self.is_cuda or self.is_rocm:
            # Only run cuda/rocm specific tests on GPU machines
            self._env_updates["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "cuda"
        elif self.is_xpu:
            self._env_updates["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "xpu"
            self._env_updates["PYTHON_TEST_EXTRA_OPTION"] = "--xpu"

        # Crossref tests
        if "crossref" in self.test_config:
            self._env_updates["PYTORCH_TEST_WITH_CROSSREF"] = "1"

    def _setup_valgrind(self) -> None:
        """Configure Valgrind settings."""
        # Default: enable valgrind
        self._env_updates["VALGRIND"] = "ON"

        # Disable valgrind for specific build environments
        disable_valgrind_patterns = [
            "clang9",  # Miscompiles std::optional<c10::SymInt>
            "xpu",
            "rocm",  # Regression in ROCm 6.0 on MI50
            "aarch64",  # TODO: revisit once CI is stabilized
            "s390x",  # Additional warnings on s390x
        ]

        for pattern in disable_valgrind_patterns:
            if pattern in self.build_environment:
                self._env_updates["VALGRIND"] = "OFF"
                break

    def _setup_sanitizers(self) -> None:
        """Configure ASAN/UBSAN settings for sanitizer builds."""
        if not self.is_asan:
            return

        # Base ASAN options
        asan_options = [
            "detect_leaks=0",
            "symbolize=1",
            "detect_stack_use_after_return=true",
            "strict_init_order=true",
            "detect_odr_violation=1",
            "detect_container_overflow=0",
            "check_initialization_order=true",
            "debug=true",
        ]

        # Additional CUDA-specific ASAN option
        if self.is_cuda:
            asan_options.append("protect_shadow_gap=0")

        self._env_updates["ASAN_OPTIONS"] = ":".join(asan_options)
        self._env_updates["UBSAN_OPTIONS"] = (
            f"print_stacktrace=1:suppressions={os.getcwd()}/ubsan.supp"
        )
        self._env_updates["PYTORCH_TEST_WITH_ASAN"] = "1"
        self._env_updates["PYTORCH_TEST_WITH_UBSAN"] = "1"
        self._env_updates["ASAN_SYMBOLIZER_PATH"] = (
            "/usr/lib/llvm-18/bin/llvm-symbolizer"
        )
        self._env_updates["TORCH_USE_RTLD_GLOBAL"] = "1"

        # Disable valgrind for ASAN builds
        self._env_updates["VALGRIND"] = "OFF"

    def _setup_cpu_capability(self) -> None:
        """Configure CPU capability settings for specific test configs."""
        if self.test_config == "nogpu_NO_AVX2":
            self._env_updates["ATEN_CPU_CAPABILITY"] = "default"
        elif self.test_config == "nogpu_AVX512":
            self._env_updates["ATEN_CPU_CAPABILITY"] = "avx2"

    def _setup_legacy_driver(self) -> None:
        """Configure legacy NVIDIA driver settings."""
        if self.test_config == "legacy_nvidia_driver":
            self._env_updates["USE_LEGACY_DRIVER"] = "1"

    # =========================================================================
    # Build Environment Property Checks
    # =========================================================================

    @property
    def is_cuda(self) -> bool:
        """Check if this is a CUDA build."""
        return "cuda" in self.build_environment

    @property
    def is_rocm(self) -> bool:
        """Check if this is a ROCm build."""
        return "rocm" in self.build_environment

    @property
    def is_xpu(self) -> bool:
        """Check if this is an XPU (Intel GPU) build."""
        return "xpu" in self.build_environment

    @property
    def is_asan(self) -> bool:
        """Check if this is an ASAN (Address Sanitizer) build."""
        return "asan" in self.build_environment

    @property
    def is_debug(self) -> bool:
        """Check if this is a debug build."""
        return "-debug" in self.build_environment

    @property
    def is_bazel(self) -> bool:
        """Check if this is a Bazel build."""
        return "-bazel-" in self.build_environment or "bazel" in self.test_config

    @property
    def is_vulkan(self) -> bool:
        """Check if this is a Vulkan build."""
        return "vulkan" in self.build_environment

    @property
    def is_libtorch(self) -> bool:
        """Check if this is a libtorch build."""
        return "libtorch" in self.build_environment

    @property
    def is_s390x(self) -> bool:
        """Check if this is an s390x build."""
        return "s390x" in self.build_environment

    @property
    def is_aarch64(self) -> bool:
        """Check if this is an aarch64 build."""
        return "aarch64" in self.build_environment

    # =========================================================================
    # Test Config Property Checks
    # =========================================================================

    @property
    def is_distributed_test(self) -> bool:
        """Check if running distributed tests."""
        return self.test_config == "distributed"

    @property
    def is_inductor_test(self) -> bool:
        """Check if running inductor tests."""
        return "inductor" in self.test_config

    @property
    def is_dynamo_test(self) -> bool:
        """Check if running dynamo tests."""
        return "dynamo" in self.test_config

    @property
    def is_benchmark_test(self) -> bool:
        """Check if running benchmark tests."""
        return any(
            x in self.test_config
            for x in ["torchbench", "huggingface", "timm", "perf"]
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

    def set(self, key: str, value: str) -> None:
        """Set an additional environment variable."""
        self._env_updates[key] = value

    def unset(self, key: str) -> None:
        """Remove an environment variable from updates."""
        self._env_updates.pop(key, None)
