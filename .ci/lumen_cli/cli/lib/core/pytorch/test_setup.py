"""
Test setup helpers for PyTorch CI tests.

This module provides test-specific setup configurations that are NOT pure
environment variables. These are setup helpers used during test execution.

For environment variable configuration, see env_config.py.
"""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def _get_env_str(key: str, default: str = "") -> str:
    """Get environment variable as string."""
    return os.environ.get(key, default)


@dataclass
class CPUAffinityConfig:
    """
    Configuration for CPU affinity settings used in benchmark tests.

    This handles the complex setup of jemalloc, Intel OpenMP, and CPU core
    allocation for performance-sensitive tests.
    """

    _env_updates: dict[str, str] = field(default_factory=dict, repr=False)
    _taskset_cmd: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Compute CPU affinity settings."""
        self._env_updates = {}
        self._setup_jemalloc()
        self._setup_openmp()
        self._setup_cpu_cores()
        self._setup_taskset()

    def _setup_jemalloc(self) -> None:
        """Configure jemalloc preloading."""
        jemalloc_lib = self._find_library("libjemalloc.so.2", "/usr/lib")
        if jemalloc_lib:
            self._add_to_ld_preload(jemalloc_lib)
            self._env_updates["MALLOC_CONF"] = (
                "oversize_threshold:1,background_thread:true,"
                "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
            )

    def _setup_openmp(self) -> None:
        """Configure Intel OpenMP for x86 systems."""
        import platform

        if platform.machine() == "aarch64":
            # Skip Intel OpenMP for ARM
            return

        # Find Intel OpenMP library
        python_bin = shutil.which("python")
        if python_bin:
            iomp_path = Path(python_bin).parent.parent / "lib" / "libiomp5.so"
            if iomp_path.exists():
                self._add_to_ld_preload(str(iomp_path))
                self._env_updates["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
                self._env_updates["KMP_BLOCKTIME"] = "1"

    def _setup_cpu_cores(self) -> None:
        """Calculate and set OMP_NUM_THREADS based on available cores."""
        import platform

        try:
            # Use nproc to account for cgroups (Linux-specific)
            if sys.platform == "linux":
                cpus = int(subprocess.check_output(["nproc"]).decode().strip())

                # Get threads per core from lscpu
                lscpu_output = subprocess.check_output(["lscpu"]).decode()
                thread_per_core = 1
                for line in lscpu_output.split("\n"):
                    if "Thread(s) per core:" in line:
                        thread_per_core = int(line.split(":")[1].strip())
                        break

                cores = cpus // thread_per_core

                # Limit to 16 cores on aarch64 for performance runs
                if platform.machine() == "aarch64" and cores > 16:
                    cores = 16
            else:
                # Fallback for non-Linux systems (macOS, Windows)
                cores = os.cpu_count() or 4

            self._env_updates["OMP_NUM_THREADS"] = str(cores)

        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            logger.warning("Failed to detect CPU cores: %s", e)
            # Fallback to os.cpu_count()
            cores = os.cpu_count() or 4
            self._env_updates["OMP_NUM_THREADS"] = str(cores)

    def _setup_taskset(self) -> None:
        """Setup taskset command for CPU pinning."""
        # taskset and sched_getaffinity are Linux-specific
        if sys.platform != "linux":
            self._taskset_cmd = ""
            return

        try:
            # Get CPU affinity range from Python
            affinity = os.sched_getaffinity(0)
            start_cpu = min(affinity)

            # Get threads per core
            lscpu_output = subprocess.check_output(["lscpu"]).decode()
            thread_per_core = 1
            for line in lscpu_output.split("\n"):
                if "Thread(s) per core:" in line:
                    thread_per_core = int(line.split(":")[1].strip())
                    break

            # Leave one physical CPU for other tasks
            end_cpu = max(affinity) - thread_per_core

            self._taskset_cmd = f"taskset -c {start_cpu}-{end_cpu}"

        except (OSError, subprocess.CalledProcessError, AttributeError) as e:
            logger.warning("Failed to setup taskset: %s", e)
            self._taskset_cmd = ""

    def _add_to_ld_preload(self, lib_path: str) -> None:
        """Add a library to LD_PRELOAD."""
        current = self._env_updates.get("LD_PRELOAD", os.environ.get("LD_PRELOAD", ""))
        if current:
            self._env_updates["LD_PRELOAD"] = f"{lib_path}:{current}"
        else:
            self._env_updates["LD_PRELOAD"] = lib_path

    def _find_library(self, name: str, search_path: str) -> Optional[str]:
        """Find a library file in the given search path."""
        try:
            result = subprocess.run(
                ["find", search_path, "-name", name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("Failed to find library %s: %s", name, e)
        return None

    @property
    def taskset(self) -> str:
        """Return the taskset command prefix, or empty string if not available."""
        return self._taskset_cmd

    def apply(self) -> None:
        """Apply CPU affinity environment variables to current process."""
        os.environ.update(self._env_updates)

    def as_dict(self) -> dict[str, str]:
        """Return environment dict for subprocess."""
        return {**os.environ, **self._env_updates}

    def get_updates(self) -> dict[str, str]:
        """Return only the computed environment variable updates."""
        return self._env_updates.copy()


@dataclass
class InductorTestConfig:
    """
    Configuration specific to inductor tests.

    Handles inductor-specific environment variables like TORCHINDUCTOR_CPP_WRAPPER,
    TORCHINDUCTOR_CUTLASS_DIR, etc.
    """

    test_config: Optional[str] = None
    _env_updates: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Resolve test_config from env if not provided
        if self.test_config is None:
            self.test_config = _get_env_str("TEST_CONFIG", "default")

        self._env_updates = {}
        self._setup_cpp_wrapper()
        self._setup_cutlass()
        self._setup_max_autotune()

    def _setup_cpp_wrapper(self) -> None:
        """Enable C++ wrapper for inductor if configured."""
        if "inductor_cpp_wrapper" in self.test_config:
            self._env_updates["TORCHINDUCTOR_CPP_WRAPPER"] = "1"

    def _setup_cutlass(self) -> None:
        """Configure CUTLASS directory for cutlass backend tests."""
        if "cutlass" in self.test_config:
            cutlass_dir = Path("./third_party/cutlass").resolve()
            if cutlass_dir.exists():
                self._env_updates["TORCHINDUCTOR_CUTLASS_DIR"] = str(cutlass_dir)

    def _setup_max_autotune(self) -> None:
        """Enable max autotune for specific test configs."""
        if "max_autotune" in self.test_config:
            self._env_updates["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"

    def apply(self) -> None:
        """Apply inductor-specific environment variables."""
        os.environ.update(self._env_updates)

    def get_updates(self) -> dict[str, str]:
        """Return the computed environment variable updates."""
        return self._env_updates.copy()


@dataclass
class XPUTestSetup:
    """
    Setup for Intel XPU (GPU) tests.

    Handles sourcing Intel oneAPI environment scripts and XPU-specific settings.
    """

    _source_scripts: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._source_scripts = []
        self._find_oneapi_scripts()

    def _find_oneapi_scripts(self) -> None:
        """Identify Intel oneAPI scripts that need to be sourced."""
        oneapi_scripts = [
            "/opt/intel/oneapi/compiler/latest/env/vars.sh",
            "/opt/intel/oneapi/umf/latest/env/vars.sh",
            "/opt/intel/oneapi/ccl/latest/env/vars.sh",
            "/opt/intel/oneapi/mpi/latest/env/vars.sh",
            "/opt/intel/oneapi/pti/latest/env/vars.sh",
        ]

        for script in oneapi_scripts:
            if Path(script).exists():
                self._source_scripts.append(script)

    def get_source_commands(self) -> list[str]:
        """Return list of source commands for oneAPI scripts."""
        return [f"source {script}" for script in self._source_scripts]

    def get_setup_script(self) -> str:
        """Return a shell script snippet to source all oneAPI scripts."""
        return "\n".join(self.get_source_commands())


@dataclass
class ROCmTestSetup:
    """Setup for AMD ROCm tests."""

    @staticmethod
    def get_benchmark_subdir() -> str:
        """Return the subdirectory for ROCm benchmark results."""
        return "rocm/"
