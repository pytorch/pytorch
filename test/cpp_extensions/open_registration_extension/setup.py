import distutils.command.clean
import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


PACKAGE_NAME = "pytorch_openreg"
version = 1.0

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "pytorch_openreg/csrc"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove pytorch_openreg extension
        for path in (ROOT_DIR / "pytorch_openreg").glob("**/*.so"):
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        vc_version = os.getenv("VCToolsVersion", "")
        if vc_version.startswith("14.16."):
            CXX_FLAGS = ["/sdl"]
        else:
            CXX_FLAGS = ["/sdl", "/permissive-"]
    elif platform.machine() == "s390x":
        # no -Werror on s390x due to newer compiler
        CXX_FLAGS = {"cxx": ["-g", "-Wall"]}
    else:
        CXX_FLAGS = {"cxx": ["-g", "-Wall", "-Werror"]}

    sources = list(CSRS_DIR.glob("*.cpp"))

    # Note that we always compile with debug info
    ext_modules = [
        CppExtension(
            name="pytorch_openreg._C",
            sources=sorted(str(s) for s in sources),
            include_dirs=[CSRS_DIR],
            extra_compile_args=CXX_FLAGS,
        )
    ]

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="PyTorch Core Team",
        description="Example for PyTorch out of tree registration",
        packages=find_packages(exclude=("test",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.dylib", "*.so"]},
        install_requires=[
            "torch",
        ],
        ext_modules=ext_modules,
        python_requires=">=3.8",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
