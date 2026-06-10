#!/usr/bin/env python3
"""Build a PyTorch wheel on Windows.

Expects vcvarsall env (PATH/LIB/INCLUDE), GPU env (USE_CUDA, CUDA_PATH,
TORCH_CUDA_ARCH_LIST, ...), and pip build deps to already be set up by
the sibling build_env_setup.py + build_install_deps.py + the parent
bash wrapper. This script is the wheel-build step proper -- the
Windows-side analog of `.ci/manywheel/build_wheel.py`.

The Linux manywheel pipeline follows the build with a repair_wheel.py
step that patches ELF RPATHs and bundles native deps. Windows doesn't
need that: DLL discovery uses PATH / AddDllDirectory rather than RPATH,
and the CUDA DLLs ship inside torch/lib/ already (torch/__init__.py
adds that directory to the dll search path at import time).

Usage: build_wheel.py <output_dir>
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
