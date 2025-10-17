import distutils.command.clean
import shutil
from pathlib import Path

import torch

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    IS_MACOS,
)


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "libtorch_agnostic" / "csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agnostic").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agnostic.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always"],
    }
    extra_link_args = []

    # Note that adding this flag does not mean extension's parallel_for will
    # always use OPENMP path, OpenMP path will only be used if (1) AND (2)
    # (1) libtorch was built with OpenMP
    # (2) extension compiles and links with -fopenmp
    if IS_MACOS:
        extra_compile_args["cxx"].extend(["-Xclang", "-fopenmp"])
        extra_link_args.append("-lomp")
    else:
        extra_compile_args["cxx"].extend(["-fopenmp"])
        extra_link_args.append("-fopenmp")

    extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        extra_compile_args["cxx"].append("-DLAE_USE_CUDA")
        extension = CUDAExtension

    sources = list(CSRC_DIR.glob("**/*.cpp"))

    return [
        extension(
            "libtorch_agnostic._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


setup(
    name="libtorch_agnostic",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agnostic": ["*.dll", "*.dylib", "*.so"]},
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
