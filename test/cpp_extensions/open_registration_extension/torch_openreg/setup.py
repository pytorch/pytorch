import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
RUN_BUILD_DEPS = any(arg in {"clean", "dist_info"} for arg in sys.argv)


def check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


if "CMAKE_BUILD_TYPE" not in os.environ:
    if check_env_flag("DEBUG"):
        os.environ["CMAKE_BUILD_TYPE"] = "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        os.environ["CMAKE_BUILD_TYPE"] = "RelWithDebInfo"
    else:
        os.environ["CMAKE_BUILD_TYPE"] = "Release"


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


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
        "--config",
        os.environ["CMAKE_BUILD_TYPE"],
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_openreg/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_openreg")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    if not RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = []
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            "-fno-strict-aliasing",
        ]

    if os.environ["CMAKE_BUILD_TYPE"] == "Debug":
        if IS_WINDOWS:
            extra_compile_args += ["/Z7"]
            extra_link_args += ["/DEBUG:FULL"]
        else:
            extra_compile_args += ["-O0", "-g"]
            extra_link_args += ["-O0", "-g"]
    elif os.environ["CMAKE_BUILD_TYPE"] == "RelWithDebInfo":
        if IS_WINDOWS:
            extra_compile_args += ["/Z7"]
            extra_link_args += ["/DEBUG:FULL"]
        else:
            extra_compile_args += ["-g"]
            extra_link_args += ["-g"]

    ext_modules = [
        Extension(
            name="torch_openreg._C",
            sources=["torch_openreg/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_openreg/lib")],
            extra_link_args=[*make_relative_rpath_args("lib")],
        )
    ]

    package_data = {
        "torch_openreg": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
    }

    setup(
        packages=find_packages(),
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
