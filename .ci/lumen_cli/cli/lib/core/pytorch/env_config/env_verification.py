"""
Build verification methods for PyTorch test configuration.

This module contains all _verify_* methods that check build configuration
correctness (ASAN, debug assertions, CUDA initialization, etc.).
"""

import logging
import subprocess

from cli.lib.common.utils import run_python_code


logger = logging.getLogger(__name__)


class BuildVerificationMixin:
    """Mixin class containing build verification methods."""

    build_environment: str
    test_config: str

    # These properties are expected from BuildEnvironmentProperties
    is_asan: bool
    is_debug: bool
    is_bazel: bool
    is_cuda: bool
    is_rocm: bool

    def as_dict(self) -> dict[str, str]:
        """Return complete environment dict (implemented in main class)."""
        raise NotImplementedError

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
            result = run_python_code(
                "import torch; print(torch.__version__, torch.version.git_version)",
                cwd=test_dir,
                env=env,
                capture_output=True,
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
        test_code = "import torch; torch.rand(2, 2, device='cuda')"
        result = run_python_code(
            test_code,
            cwd=test_dir,
            env=env,
            check=False,
            capture_output=True,
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
            result = run_python_code(
                f"import torch; {crash_code}",
                cwd=test_dir,
                env=env,
                check=False,
                capture_output=True,
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
        debug_test_code = "import torch; torch._C._crash_if_debug_asserts_fail(424242)"

        result = run_python_code(
            debug_test_code,
            cwd=test_dir,
            env=env,
            check=False,
            capture_output=True,
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
