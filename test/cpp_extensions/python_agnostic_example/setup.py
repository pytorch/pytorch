# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import glob
import os
import shutil
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
)


ROOT_DIR = Path(__file__).absolute().parent


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchao extension
        for path in (ROOT_DIR / "torchao").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "torchao.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extensions():
    if not torch.cuda.is_available():
        print(
            "PyTorch GPU support is not available. Skipping compilation of CUDA extensions"
        )
    if CUDA_HOME is None and torch.cuda.is_available():
        print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")
        print(
            "If you'd like to compile CUDA extensions locally please install the cudatoolkit from https://anaconda.org/nvidia/cuda-toolkit"
        )

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3",
            "-t=0",
        ],
    }

    sources = list(glob.glob(os.path.join(ROOT_DIR, "**/*.cpp"), recursive=True))
    cuda_sources = list(glob.glob(os.path.join(ROOT_DIR, "**/*.cu"), recursive=True))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            "torchao._C",
            sources,
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name="torchao",
    version="0.0",
    ext_modules=get_extensions(),
    description="Package for applying ao techniques to GPU models",
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=False),
        "clean": clean,
    },
    options={"bdist_wheel": {
        "py_limited_api": "cp38"
        }
    },
)
