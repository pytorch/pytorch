#!/usr/bin/env python3
"""Install build-time dependencies for a PyTorch wheel build.

Usage: build_install_deps.py <package_dir>

Environment variables:
    DESIRED_CUDA - CUDA variant; "rocm*" triggers the AMD source-rewrite step.
"""

import argparse
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


# NumPy build-time pin selected by Python version. Checked high-to-low; the
# first entry whose (major, minor) floor is satisfied wins. A plain string
# prefix ("cp31") would wrongly capture cp315, so match on the version tuple.
# Keep in sync with .ci/manywheel/build_common.sh.
NUMPY_PINS: list[tuple[tuple[int, int], str]] = [
    ((3, 15), "2.5.1"),
    ((3, 14), "2.3.4"),
    ((3, 10), "2.1.0"),
]
DEFAULT_NUMPY = "2.0.2"


def retry(cmd: list[str], delays: tuple[int, ...] = (1, 2, 4, 8)) -> None:
    """Run cmd, retrying with backoff on failure (mirrors the shell retry helper)."""
    last_rc = 0
    for delay in (0, *delays):
        if delay:
            time.sleep(delay)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        last_rc = result.returncode
    sys.exit(last_rc)


def pip_install(*args: str) -> None:
    retry([sys.executable, "-m", "pip", "install", *args])


def numpy_pin() -> str:
    version = sys.version_info[:2]
    for floor, pin in NUMPY_PINS:
        if version >= floor:
            return pin
    return DEFAULT_NUMPY


def is_rocm_py315() -> bool:
    return sys.version_info[:2] == (3, 15) and "rocm" in os.environ.get(
        "DESIRED_CUDA", ""
    )


def prefer_target_python_pkgconfig() -> None:
    """Make ``pkg-config python3`` resolve the interpreter we are building for.

    cp315 has no numpy wheel, so numpy is built from source. On the ROCm
    manylinux image the distro's system ``python3-devel`` (Python 3.6) has its
    ``python3.pc`` on pkg-config's default search path, so numpy's meson Cython
    check resolves ``pkg-config python3`` to 3.6 and fails with "Cython requires
    Python 3.8+". Prepend this interpreter's own pkgconfig dir so the correct
    Python wins (pkg-config searches PKG_CONFIG_PATH before its default libdir).

    Scoped to ROCm cp315: the CUDA/XPU/CPU images have no such stray python3.pc,
    so their from-source numpy build already resolves correctly.
    """
    libpc = sysconfig.get_config_var("LIBPC")
    if libpc and os.path.isfile(os.path.join(libpc, "python3.pc")):
        existing = os.environ.get("PKG_CONFIG_PATH", "")
        os.environ["PKG_CONFIG_PATH"] = (
            f"{libpc}{os.pathsep}{existing}" if existing else libpc
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    # ROCm cp315 builds numpy from source (no cp315 wheel); redirect
    # `pkg-config python3` to the interpreter we're building for so the ROCm
    # image's system 3.6 python3.pc can't hijack it. No-op elsewhere.
    if is_rocm_py315():
        prefer_target_python_pkgconfig()
    pip_install("-qU", "-r", "requirements-build.txt")
    # The CUPTI field-id codegen (tools/gen_cupti_stubs.py) parses cupti_activity.h with
    # libclang's python bindings. Install libclang only when a sufficiently-new CUPTI header
    # is actually resolvable (find_cupti_header applies the CUPTI_API_VERSION floor) -- so
    # non-13.x / CPU / ROCm / XPU builds, which have no such header, don't pull it in. Run in
    # a subprocess with cwd on sys.path (we chdir'd to the repo root above) so the import of
    # tools.setup_helpers.cupti resolves.
    header_available = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tools.setup_helpers.cupti import find_cupti_header as f;"
            " sys.exit(0 if f() else 1)",
        ]
    )
    # Skip the (heavy, version-matched) libclang wheel when LIBCLANG_PATH already points the
    # codegen at a libclang.so -- that env is expected to supply the clang bindings itself.
    if header_available.returncode == 0 and not os.environ.get("LIBCLANG_PATH"):
        pip_install("-q", "libclang")
    # Skip when sharing build/ across Pythons in build_all.sh -- the per-Python
    # bits (libtorch_python, _C.so) are invalidated by tools/setup_helpers/cmake.py.
    if not os.environ.get("SKIP_SETUP_CLEAN"):
        subprocess.run([sys.executable, "-m", "spin", "clean"], check=True)
    pip_install("-q", "-r", "requirements.txt")
    pip_install("-q", "--pre", f"numpy=={numpy_pin()}")
    # auditwheel repacks the manywheel with a valid ZIP64 record for >4GB ROCm
    # wheels (pypa/wheel#692); imported by repair_wheel.py. CD-only, so it is
    # installed here rather than in requirements.txt, and pinned for
    # reproducible binary builds.
    pip_install("-q", "auditwheel==6.4.2")

    if "rocm" in os.environ.get("DESIRED_CUDA", ""):
        print(f"Running build_amd.py at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        subprocess.run([sys.executable, "tools/amd_build/build_amd.py"], check=True)


if __name__ == "__main__":
    main()
