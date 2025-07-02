import os

from setuptools import find_packages, setup, Extension
from distutils.command.clean import clean
from torch.utils.cpp_extension import BuildExtension
from sysconfig import get_paths

import subprocess
import multiprocessing
import shutil

PACKAGE_NAME = "torch_openreg"
version = 1.0

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pytorch_dir():
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    output_lib_path = os.path.join(BASE_DIR, "torch_openreg")
    os.makedirs(output_lib_path, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX=" + os.path.realpath(output_lib_path),
        "-DPYTHON_INCLUDE_DIR=" + get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--",
    ]

    build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


ext_modules = [
    Extension(
        name="torch_openreg._C",
        sources=["torch_openreg/csrc/stub.c"],
        extra_compile_args=["-g", "-Wall", "-Werror"],
        libraries=["torch_openreg_python"],
        library_dirs=[os.path.join(BASE_DIR, "torch_openreg/lib")],
        extra_link_args=["-Wl,-rpath,$ORIGIN/lib"],
        define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")],
    )
]


class BuildExt(BuildExtension):
    def run(self):
        build_deps()

        super().run()


class BuildClean(clean):
    def run(self):
        self.clean_egginfo()

        build_dir = os.path.join(BASE_DIR, "build")
        if os.path.exists(build_dir) and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)

    def clean_egginfo(self):
        egginfo_dir = os.path.join(BASE_DIR, "torch_openreg.egg-info")

        if os.path.exists(egginfo_dir) and os.path.isdir(egginfo_dir):
            shutil.rmtree(egginfo_dir)


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
        "build_ext": BuildExt,  # type: ignore[misc]
        "clean": BuildClean,  # type: ignore[misc]
    },
)
