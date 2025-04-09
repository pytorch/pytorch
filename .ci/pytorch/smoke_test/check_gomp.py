import ctypes
import os
import sys
from pathlib import Path


def get_gomp_thread():
    """
    Retrieves the maximum number of OpenMP threads after loading the `libgomp.so.1` library
    and the `libtorch_cpu.so` library. It then queries the
    maximum number of threads available for OpenMP parallel regions using the
    `omp_get_max_threads` function.

    Returns:
        int: The maximum number of OpenMP threads available.

    Notes:
        - The function assumes the default path for `libgomp.so.1` on AlmaLinux OS.
        - The path to `libtorch_cpu.so` is constructed based on the Python executable's
          installation directory.
        - This function is specific to environments where PyTorch and OpenMP are used
          together and may require adjustments for other setups.
    """
    python_path = Path(sys.executable).resolve()
    python_prefix = (
        python_path.parent.parent
    )  # Typically goes to the Python installation root

    # Get the additional ABI flags (if any); it may be an empty string.
    abiflags = getattr(sys, "abiflags", "")

    # Construct the Python directory name correctly (e.g., "python3.13t").
    python_version = (
        f"python{sys.version_info.major}.{sys.version_info.minor}{abiflags}"
    )

    libtorch_cpu_path = (
        python_prefix
        / "lib"
        / python_version
        / "site-packages"
        / "torch"
        / "lib"
        / "libtorch_cpu.so"
    )

    # use the default gomp path of AlmaLinux OS
    libgomp_path = "/usr/lib64/libgomp.so.1"

    os.environ["GOMP_CPU_AFFINITY"] = "0-3"

    libgomp = ctypes.CDLL(libgomp_path)
    libgomp = ctypes.CDLL(libtorch_cpu_path)

    libgomp.omp_get_max_threads.restype = ctypes.c_int
    libgomp.omp_get_max_threads.argtypes = []

    omp_max_threads = libgomp.omp_get_max_threads()
    return omp_max_threads


def main():
    omp_max_threads = get_gomp_thread()
    print(
        f"omp_max_threads after loading libgomp.so and libtorch_cpu.so: {omp_max_threads}"
    )
    if omp_max_threads == 1:
        raise RuntimeError(
            "omp_max_threads is 1. Check whether libgomp.so is loaded twice."
        )


if __name__ == "__main__":
    main()
