# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, SyclExtension


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "python_agnostic" / "csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "python_agnostic").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "python_agnostic.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always"],
    }

    extensions = []
    if torch.cuda.is_available():
        cuda_sources = list(CSRC_DIR.glob("**/*.cu"))

        extensions.append(
            CUDAExtension(
                "python_agnostic.cuda",
                sources=sorted(str(s) for s in cuda_sources),
                py_limited_api=True,
                extra_compile_args=extra_compile_args,
                extra_link_args=[],
            )
        )
    if torch.xpu.is_available():
        sycl_sources = list(CSRC_DIR.glob("**/*.sycl"))

        extensions.append(
            SyclExtension(
                "python_agnostic.sycl",
                sources=sorted(str(s) for s in sycl_sources),
                py_limited_api=True,
                extra_compile_args=extra_compile_args,
                extra_link_args=[],
            )
        )
    return extensions


setup(
    name="python_agnostic",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of python agnostic extension",
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
