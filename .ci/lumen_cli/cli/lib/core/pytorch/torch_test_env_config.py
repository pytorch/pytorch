"""
Environment configuration for PyTorch CI tests.

This module provides centralized environment variable management for PyTorch tests,
replacing the scattered environment setup in test.sh with a structured, testable approach.

This module focuses ONLY on environment variable configuration.
For test-specific setup (CPU affinity, inductor config, etc.), see test_setup.py.

Usage:
    from cli.lib.core.pytorch.env_config import TestEnvironment

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

    # Or pass to subprocess
    run_command("python test/run_test.py ...", env=env.as_dict())
"""

import logging
import os
import shutil
import subprocess
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
class PytorchTestEnvironment:
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
        self._setup_build_dirs()
        self._setup_torch_install_dirs()
        self._setup_device_visibility()
        self._setup_test_flags()
        self._setup_valgrind()
        self._setup_sanitizers()
        self._setup_cpu_capability()
        self._setup_legacy_driver()
        self._setup_cuda_arch()

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

    def _setup_build_dirs(self) -> None:
        """Configure build directory paths."""
        build_dir = "build"
        self._env_updates["BUILD_DIR"] = build_dir
        self._env_updates["BUILD_RENAMED_DIR"] = "build_renamed"
        self._env_updates["BUILD_BIN_DIR"] = f"{build_dir}/bin"

    def _setup_torch_install_dirs(self) -> None:
        """
        Configure Torch installation directory paths.

        Determines the torch install location by querying Python's site-packages,
        then sets paths for bin, lib, and test directories.
        """
        try:
            result = subprocess.run(
                ["python", "-c", "import site; print(site.getsitepackages()[0])"],
                capture_output=True,
                text=True,
                check=True,
            )
            site_packages = result.stdout.strip()
            torch_install_dir = f"{site_packages}/torch"

            self._env_updates["TORCH_INSTALL_DIR"] = torch_install_dir
            self._env_updates["TORCH_BIN_DIR"] = f"{torch_install_dir}/bin"
            self._env_updates["TORCH_LIB_DIR"] = f"{torch_install_dir}/lib"
            self._env_updates["TORCH_TEST_DIR"] = f"{torch_install_dir}/test"
        except subprocess.CalledProcessError:
            logger.warning("Failed to determine torch install directory from Python")

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

        # Set LD_PRELOAD to the ASAN runtime library
        if shutil.which("clang"):
            try:
                result = subprocess.run(
                    ["clang", "--print-file-name=libclang_rt.asan-x86_64.so"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ld_preload = result.stdout.strip()
                if ld_preload:
                    self._env_updates["LD_PRELOAD"] = ld_preload
            except subprocess.CalledProcessError:
                logger.warning("Failed to get ASAN runtime library path from clang")

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

    def _setup_cuda_arch(self) -> None:
        """
        Detect and configure CUDA architecture.

        Sets TORCH_CUDA_ARCH_LIST based on:
        - nvidia-smi query if available (gets actual GPU compute capability)
        - Default value of 8.0 for nogpu tests where nvidia-smi isn't available
        """
        if not self.is_cuda:
            return

        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines = result.stdout.strip().splitlines()
                if len(lines) >= 2:
                    cuda_arch = lines[-1].strip()
                    self._env_updates["TORCH_CUDA_ARCH_LIST"] = cuda_arch
            except subprocess.CalledProcessError:
                logger.warning("Failed to query CUDA architecture from nvidia-smi")
        elif "nogpu" in self.test_config:
            # There won't be nvidia-smi in nogpu tests, so set to default minimum supported value
            self._env_updates["TORCH_CUDA_ARCH_LIST"] = "8.0"

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
            x in self.test_config for x in ["torchbench", "huggingface", "timm", "perf"]
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def apply(self) -> None:
        """Apply all computed environment variables to the current process."""
        os.environ.update(self._env_updates)
        logger.info("Applied %d environment variables", len(self._env_updates))

    def verify_build_configuration(self, test_dir: str = "test") -> None:
        """
        Verify that the build configuration is correct by running diagnostic tests.

        This runs PyTorch's built-in test functions that verify:
        - ASAN/UBSAN sanitizers are working (for ASAN builds)
        - Debug assertions are working (for debug builds)
        - Debug assertions are disabled (for non-debug, non-bazel builds)

        Args:
            test_dir: Directory to run the tests from (default: "test")

        Raises:
            RuntimeError: If build configuration is incorrect
        """
        env = self.as_dict()

        # Skip bazel builds entirely - torch isn't available there yet
        if self.is_bazel:
            logger.debug(
                "Skipping build verification (bazel build - torch not available)"
            )
            return

        # First, verify PyTorch is importable and print version
        logger.info("Verifying PyTorch installation...")
        try:
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    "import torch; print(torch.__version__, torch.version.git_version)",
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=test_dir,
                env=env,
            )
            logger.info("PyTorch version: %s", result.stdout.strip())
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to import PyTorch: {e.stderr}") from e

        # Verify ASAN/UBSAN configuration
        if self.is_asan:
            self._verify_sanitizer_configuration(test_dir, env)

        # Verify debug configuration
        self._verify_debug_configuration(test_dir, env)

        # Verify CUDA can be initialized for legacy_nvidia_driver
        if self.test_config == "legacy_nvidia_driver":
            self._verify_cuda_can_be_initialized(test_dir, env)

    def _verify_cuda_can_be_initialized(
        self, test_dir: str, env: dict[str, str]
    ) -> None:
        """
        Verify that CUDA can be initialized successfully.

        This creates a simple CUDA tensor to ensure the CUDA runtime
        is properly configured and the GPU is accessible.

        Only runs for CUDA builds (not ROCm, XPU, or CPU-only).
        """
        if not self.is_cuda:
            logger.debug("Skipping CUDA initialization check (not a CUDA build)")
            return

        logger.info("Verifying CUDA can be initialized...")
        test_code = "torch.rand(2, 2, device='cuda')"
        result = subprocess.run(
            ["python", "-c", f"import torch; {test_code}"],
            capture_output=True,
            text=True,
            cwd=test_dir,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"CUDA initialization failed. Unable to create tensor on CUDA device.\n"
                f"stderr: {result.stderr}"
            )

        logger.info("✓ CUDA initialized successfully")

    def _verify_sanitizer_configuration(
        self, test_dir: str, env: dict[str, str]
    ) -> None:
        """
        Verify that ASAN/UBSAN are properly configured by running intentional crash tests.

        These tests intentionally trigger memory bugs. If the sanitizers are working
        correctly, these should crash. If they don't crash, ASAN/UBSAN is misconfigured.
        """
        crash_tests = [
            (
                "torch._C._crash_if_csrc_asan(3)",
                "ASAN crash test (csrc)",
            ),
            (
                "torch._C._crash_if_vptr_ubsan()",
                "UBSAN vptr crash test",
            ),
            (
                "torch._C._crash_if_aten_asan(3)",
                "ASAN crash test (aten)",
            ),
        ]

        logger.info(
            "Running sanitizer verification tests "
            "(these are expected to crash if ASAN/UBSAN is configured correctly)..."
        )

        for crash_code, description in crash_tests:
            result = subprocess.run(
                ["python", "-c", f"import torch; {crash_code}"],
                capture_output=True,
                cwd=test_dir,
                env=env,
            )

            if result.returncode == 0:
                raise RuntimeError(
                    f"ASAN/UBSAN misconfigured: {description} did not crash. "
                    "The sanitizer should have detected this intentional bug."
                )

            logger.info(
                "✓ %s: crashed as expected (exit code %d)",
                description,
                result.returncode,
            )

        logger.info("Sanitizer configuration verified successfully")

    def _verify_debug_configuration(self, test_dir: str, env: dict[str, str]) -> None:
        """
        Verify that debug assertions are correctly enabled/disabled.

        - In debug mode: _crash_if_debug_asserts_fail should crash
        - In non-debug mode: _crash_if_debug_asserts_fail should pass (noop)
        """
        debug_test_code = "torch._C._crash_if_debug_asserts_fail(424242)"

        result = subprocess.run(
            ["python", "-c", f"import torch; {debug_test_code}"],
            capture_output=True,
            cwd=test_dir,
            env=env,
        )

        if self.is_debug:
            # Debug mode: expect the assertion to fail (crash)
            logger.info(
                "Debug mode detected (%s). Verifying debug assertions are enabled...",
                self.build_environment,
            )
            if result.returncode == 0:
                raise RuntimeError(
                    "Debug build misconfigured: _crash_if_debug_asserts_fail did not crash. "
                    "Debug assertions should be enabled in debug builds."
                )
            logger.info(
                "✓ Debug assertion test: crashed as expected (exit code %d)",
                result.returncode,
            )
        else:
            # Non-debug mode: expect the assertion to pass (noop)
            logger.info(
                "Non-debug mode detected (%s). Verifying debug assertions are disabled...",
                self.build_environment,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Non-debug build misconfigured: _crash_if_debug_asserts_fail crashed. "
                    "Debug assertions should be disabled in non-debug builds. "
                    f"stderr: {result.stderr.decode() if result.stderr else 'N/A'}"
                )
            logger.info("✓ Debug assertion test: passed as expected (noop)")

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
