import distutils.command.clean
import shutil
from pathlib import Path

import torch

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "ops_test" / "csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "ops_test").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "ops_test.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always"],
    }

    extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        extension = CUDAExtension

    sources = list(CSRC_DIR.glob("**/*.cpp"))

    return [
        extension(
            "ops_test._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name="ops_test",
    version="0.0",
    author="PyTorch Core Team",
    description="Test extension for ops.h",
    packages=find_packages(exclude=("test",)),
    package_data={"ops_test": ["*.dll", "*.dylib", "*.so"]},
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
