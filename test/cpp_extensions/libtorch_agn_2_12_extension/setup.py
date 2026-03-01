import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    IS_WINDOWS,
)


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"

# Include csrc from previous versions for forward compatibility testing
PREV_CSRC_DIRS = [
    ROOT_DIR.parent / "libtorch_agn_2_9_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_10_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_11_extension" / "csrc",
]


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_12").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_12.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": [
            "-DTORCH_TARGET_VERSION=0x020c000000000000",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_12",
        ],
    }
    if not IS_WINDOWS:
        extra_compile_args["cxx"].append("-fdiagnostics-color=always")

    # Collect sources from this version and all previous versions
    sources = list(CSRC_DIR.glob("**/*.cpp"))
    for prev_dir in PREV_CSRC_DIRS:
        sources.extend(prev_dir.glob("**/*.cpp"))

    extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        extra_compile_args["cxx"].append("-DLAE_USE_CUDA")
        extra_compile_args["nvcc"] = [
            "-O2",
            "-DUSE_CUDA",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_12",
        ]
        extension = CUDAExtension
        sources.extend(CSRC_DIR.glob("**/*.cu"))
        for prev_dir in PREV_CSRC_DIRS:
            sources.extend(prev_dir.glob("**/*.cu"))

    return [
        extension(
            "libtorch_agn_2_12._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name="libtorch_agn_2_12",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension for PyTorch 2.12+",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_12": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
