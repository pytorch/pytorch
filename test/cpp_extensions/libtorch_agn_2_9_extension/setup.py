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


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_9").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_9.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": [
            "-DTORCH_STABLE_ONLY",
            "-DTORCH_TARGET_VERSION=0x0209000000000000",
        ],
    }
    if not IS_WINDOWS:
        extra_compile_args["cxx"].append("-fdiagnostics-color=always")

    sources = list(CSRC_DIR.glob("**/*.cpp"))

    extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        extra_compile_args["cxx"].append("-DLAE_USE_CUDA")
        extra_compile_args["nvcc"] = [
            "-O2",
            "-DTORCH_TARGET_VERSION=0x0209000000000000",
        ]
        extension = CUDAExtension
        sources.extend(CSRC_DIR.glob("**/*.cu"))

    return [
        extension(
            "libtorch_agn_2_9._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name="libtorch_agn_2_9",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension for PyTorch 2.9",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_9": ["*.dll", "*.dylib", "*.so"]},
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
