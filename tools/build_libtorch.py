import argparse
import sys
from os.path import abspath, dirname


# By appending pytorch_root to sys.path, this module can import other torch
# modules even when run as a standalone script. i.e., it's okay either you
# do `python build_libtorch.py` or `python -m tools.build_libtorch`.
pytorch_root = dirname(dirname(abspath(__file__)))
sys.path.append(pytorch_root)

from tools.build_pytorch_libs import build_pytorch
from tools.setup_helpers.cmake import CMake


if __name__ == "__main__":
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description="Build libtorch")
    parser.add_argument("--rerun-cmake", action="store_true", help="rerun cmake")
    parser.add_argument(
        "--cmake-only",
        action="store_true",
        help="Stop once cmake terminates. Leave users a chance to adjust build options",
    )
    options = parser.parse_args()

    build_pytorch(
        version=None,
        cmake_python_library=None,
        build_python=False,
        rerun_cmake=options.rerun_cmake,
        cmake_only=options.cmake_only,
        cmake=CMake(),
    )
