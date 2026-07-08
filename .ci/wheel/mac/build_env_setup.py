#!/usr/bin/env python3
"""macOS arm64 build environment setup (runs once per Python before the build).

Mirrors .ci/wheel/linux/build_env_setup.py for the macOS CD wheel build. macOS
needs far less toolchain wiring than the manylinux path: there is no
CUDA/ROCm/XPU split and the heavy deps come from the runner image or Homebrew.
This script just emits the macOS build flags to the --env-out file so the
caller (build.sh) can source them into the wheel build subprocess. Without that
handoff the exports made here die with this process.

Environment variables written (to --env-out):
    OMP_PREFIX - if /opt/llvm-openmp exists it is exported so the build links
                 the conda-forge libomp (supports older macOS than the Homebrew
                 build). See .ci/wheel/mac/install_libomp.sh.
    (plus the static macOS build flags in MACOS_BUILD_ENV)
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/wheel
from _common import write_env_exports


# macOS arm64 build flags.
# USE_DISTRIBUTED=1 needs libuv, built as part of the tensorpipe submodule.
# MKLDNN/QNNPACK are off on Apple silicon.
MACOS_BUILD_ENV: dict[str, str] = {
    "TH_BINARY_BUILD": "1",
    "INSTALL_TEST": "0",
    "MACOSX_DEPLOYMENT_TARGET": "14.0",
    "USE_DISTRIBUTED": "1",
    "USE_MKLDNN": "OFF",
    "USE_QNNPACK": "OFF",
    "BUILD_TEST": "OFF",
    "USE_PYTORCH_METAL_EXPORT": "1",
    "USE_COREML_DELEGATE": "1",
}

OMP_PREFIX = Path("/opt/llvm-openmp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for build.sh to source.",
    )
    args = parser.parse_args()

    env_out = dict(MACOS_BUILD_ENV)
    # Prefer the conda-forge libomp staged at /opt/llvm-openmp; otherwise the
    # build falls back to the Homebrew libomp installed in build_install_deps.py.
    if OMP_PREFIX.is_dir():
        env_out["OMP_PREFIX"] = str(OMP_PREFIX)

    write_env_exports(env_out, args.env_out)
    print("macOS build environment configured")


if __name__ == "__main__":
    main()
