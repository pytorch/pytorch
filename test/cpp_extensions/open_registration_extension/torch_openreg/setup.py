import multiprocessing
import os
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


PACKAGE_NAME = "torch_openreg"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pytorch_dir():
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_openreg")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
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


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_openreg.egg-info", "torch_openreg/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_openreg")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


RUN_BUILD_DEPS = any(arg == "clean" for arg in sys.argv)


def main():
    if not RUN_BUILD_DEPS:
        build_deps()

    ext_modules = [
        Extension(
            name="torch_openreg._C",
            sources=["torch_openreg/csrc/stub.c"],
            extra_compile_args=["-g", "-Wall", "-Werror"],
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_openreg/lib")],
            extra_link_args=["-Wl,-rpath,$ORIGIN/lib"],
        )
    ]

    package_data = {PACKAGE_NAME: ["lib/*.so*"]}

    setup(
        name=PACKAGE_NAME,
        version="0.0.1",
        author="PyTorch Core Team",
        description="Example for PyTorch out of tree registration",
        packages=find_packages(exclude=("test",)),
        package_data=package_data,
        install_requires=[
            "torch",
        ],
        ext_modules=ext_modules,
        python_requires=">=3.8",
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
    )


if __name__ == "__main__":
    main()
