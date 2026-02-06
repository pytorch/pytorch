"""
Environment setup methods for PyTorch test configuration.

This module contains all _setup_* methods that configure environment variables
based on build environment and test configuration.
"""

import logging
import os
import re
import shutil
import subprocess

from cli.lib.common.utils import run_python_code


logger = logging.getLogger(__name__)


class EnvironmentSetupMixin:
    """Mixin class containing environment setup methods."""

    _env_updates: dict[str, str]
    build_environment: str
    test_config: str

    # These properties are expected from BuildEnvironmentProperties
    is_cuda: bool
    is_rocm: bool
    is_xpu: bool
    is_asan: bool
    is_bazel: bool

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
            result = run_python_code(
                "import site; print(site.getsitepackages()[0])",
                capture_output=True,
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
            # Disable timeout due to shard not balance for xpu
            self._env_updates["NO_TEST_TIMEOUT"] = "True"

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

    def _setup_custom_test_artifact_dir(self) -> None:
        """Configure custom test artifact build directory."""
        if self.is_bazel:
            return

        default_dir = "build/custom_test_artifacts"
        custom_dir = os.environ.get("CUSTOM_TEST_ARTIFACT_BUILD_DIR", default_dir)
        self._env_updates["CUSTOM_TEST_ARTIFACT_BUILD_DIR"] = os.path.realpath(
            custom_dir
        )

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

    def _setup_rocm_arch(self) -> None:
        """
        Detect and configure ROCm GPU architecture.

        Uses rocminfo to query GPU information and sets:
        - PYTORCH_ROCM_ARCH: The GPU architecture (e.g., gfx90a, gfx942)
        - ROCM_PATH: Path prefix for ROCm (if detected)
        """
        if not self.is_rocm:
            return

        if not shutil.which("rocminfo"):
            raise RuntimeError(
                "rocminfo not found, skipping ROCm architecture detection"
            )

        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout

            # Parse rocminfo output for GPU architecture (gfx*)
            gfx_matches = re.findall(r"Name:\s*(gfx\w+)", output)
            marketing_matches = re.findall(r"Marketing Name:\s*(.+)", output)

            if gfx_matches:
                # Use the first GPU architecture found
                rocm_arch = gfx_matches[0].strip()
                self._env_updates["PYTORCH_ROCM_ARCH"] = rocm_arch
                logger.info("Detected ROCm GPU architecture: %s", rocm_arch)

            if marketing_matches:
                logger.info(
                    "ROCm GPU Marketing Name: %s", marketing_matches[0].strip()
                )

            # Set ROCM_PATH if not already set
            if "ROCM_PATH" not in os.environ:
                self._env_updates["ROCM_PATH"] = "/opt/rocm"

        except subprocess.CalledProcessError:
            logger.warning("Failed to query ROCm architecture from rocminfo")

    def run_all_setup(self) -> None:
        """Run all environment setup methods in order."""
        self._setup_base_env()
        self._setup_build_dirs()
        self._setup_torch_install_dirs()
        self._setup_device_visibility()
        self._setup_test_flags()
        self._setup_valgrind()
        self._setup_sanitizers()
        self._setup_cpu_capability()
        self._setup_legacy_driver()
        self._setup_custom_test_artifact_dir()
        self._setup_cuda_arch()
        self._setup_rocm_arch()
