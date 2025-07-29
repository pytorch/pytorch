# Welcome to the PyTorch setup.py.
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   REL_WITH_DEB_INFO
#     build with optimizations and -g (debug symbols)
#
#   USE_CUSTOM_DEBINFO="path/to/file1.cpp;path/to/file2.cpp"
#     build with debug info only for specified files
#
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
#   USE_CUDA=0
#     disables CUDA build
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files (unless CXXFLAGS is set), in contrast to the
#     default behavior of autogoo and cmake build systems.)
#
#     A specific flag that can be used is
#     -DHAS_TORCH_SHOW_DISPATCH_TRACE
#       build with dispatch trace that can be enabled with
#       TORCH_SHOW_DISPATCH_TRACE=1 at runtime.
#
#   CC
#     the C/C++ compiler to use
#
#   CMAKE_FRESH=1
#     force a fresh cmake configuration run, ignoring the existing cmake cache
#
#   CMAKE_ONLY=1
#     run cmake and stop; do not build the project
#
# Environment variables for feature toggles:
#
#   DEBUG_CUDA=1
#     if used in conjunction with DEBUG or REL_WITH_DEB_INFO, will also
#     build CUDA kernels with -lineinfo --source-in-ptx.  Note that
#     on CUDA 12 this may cause nvcc to OOM, so this is disabled by default.
#
#   USE_CUDNN=0
#     disables the cuDNN build
#
#   USE_CUSPARSELT=0
#     disables the cuSPARSELt build
#
#   USE_CUDSS=0
#     disables the cuDSS build
#
#   USE_CUFILE=0
#     disables the cuFile build
#
#   USE_FBGEMM=0
#     disables the FBGEMM build
#
#   USE_KINETO=0
#     disables usage of libkineto library for profiling
#
#   USE_NUMPY=0
#     disables the NumPy build
#
#   BUILD_TEST=0
#     disables the test build
#
#   USE_MKLDNN=0
#     disables use of MKLDNN
#
#   USE_MKLDNN_ACL
#     enables use of Compute Library backend for MKLDNN on Arm;
#     USE_MKLDNN must be explicitly enabled.
#
#   MKLDNN_CPU_RUNTIME
#     MKL-DNN threading mode: TBB or OMP (default)
#
#   USE_STATIC_MKL
#     Prefer to link with MKL statically - Unix only
#   USE_ITT=0
#     disable use of Intel(R) VTune Profiler's ITT functionality
#
#   USE_NNPACK=0
#     disables NNPACK build
#
#   USE_DISTRIBUTED=0
#     disables distributed (c10d, gloo, mpi, etc.) build
#
#   USE_TENSORPIPE=0
#     disables distributed Tensorpipe backend build
#
#   USE_GLOO=0
#     disables distributed gloo backend build
#
#   USE_MPI=0
#     disables distributed MPI backend build
#
#   USE_SYSTEM_NCCL=0
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   USE_OPENMP=0
#     disables use of OpenMP for parallelization
#
#   USE_FLASH_ATTENTION=0
#     disables building flash attention for scaled dot product attention
#
#   USE_MEM_EFF_ATTENTION=0
#    disables building memory efficient attention for scaled dot product attention
#
#   BUILD_BINARY
#     enables the additional binaries/ build
#
#   ATEN_AVX512_256=TRUE
#     ATen AVX2 kernels can use 32 ymm registers, instead of the default 16.
#     This option can be used if AVX512 doesn't perform well on a machine.
#     The FBGEMM library also uses AVX512_256 kernels on Xeon D processors,
#     but it also has some (optimized) assembly code.
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   TORCH_CUDA_ARCH_LIST
#     specify which CUDA architectures to build for.
#     ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
#     These are not CUDA versions, instead, they specify what
#     classes of NVIDIA hardware we should generate PTX for.
#
#   TORCH_XPU_ARCH_LIST
#     specify which XPU architectures to build for.
#     ie `TORCH_XPU_ARCH_LIST="ats-m150,lnl-m"`
#
#   PYTORCH_ROCM_ARCH
#     specify which AMD GPU targets to build for.
#     ie `PYTORCH_ROCM_ARCH="gfx900;gfx906"`
#
#   ONNX_NAMESPACE
#     specify a namespace for ONNX built here rather than the hard-coded
#     one in this file; needed to build with other frameworks that share ONNX.
#
#   BLAS
#     BLAS to be used by Caffe2. Can be MKL, Eigen, ATLAS, FlexiBLAS, or OpenBLAS. If set
#     then the build will fail if the requested BLAS is not found, otherwise
#     the BLAS will be chosen based on what is found on your system.
#
#   MKL_THREADING
#     MKL threading mode: SEQ, TBB or OMP (default)
#
#   USE_ROCM_KERNEL_ASSERT=1
#     Enable kernel assert in ROCm platform
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   CUDA_HOME (Linux/OS X)
#   CUDA_PATH (Windows)
#     specify where CUDA is installed; usually /usr/local/cuda or
#     /usr/local/cuda-x.y
#   CUDAHOSTCXX
#     specify a different compiler than the system one to use as the CUDA
#     host compiler for nvcc.
#
#   CUDA_NVCC_EXECUTABLE
#     Specify a NVCC to use. This is used in our CI to point to a cached nvcc
#
#   CUDNN_LIB_DIR
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARY
#     specify where cuDNN is installed
#
#   MIOPEN_LIB_DIR
#   MIOPEN_INCLUDE_DIR
#   MIOPEN_LIBRARY
#     specify where MIOpen is installed
#
#   NCCL_ROOT
#   NCCL_LIB_DIR
#   NCCL_INCLUDE_DIR
#     specify where nccl is installed
#
#   ACL_ROOT_DIR
#     specify where Compute Library is installed
#
#   LIBRARY_PATH
#   LD_LIBRARY_PATH
#     we will search for libraries in these paths
#
#   ATEN_THREADING
#     ATen parallel backend to use for intra- and inter-op parallelism
#     possible values:
#       OMP - use OpenMP for intra-op and native backend for inter-op tasks
#       NATIVE - use native thread pool for both intra- and inter-op tasks
#
#   USE_SYSTEM_LIBS (work in progress)
#      Use system-provided libraries to satisfy the build dependencies.
#      When turned on, the following cmake variables will be toggled as well:
#        USE_SYSTEM_CPUINFO=ON
#        USE_SYSTEM_SLEEF=ON
#        USE_SYSTEM_GLOO=ON
#        BUILD_CUSTOM_PROTOBUF=OFF
#        USE_SYSTEM_EIGEN_INSTALL=ON
#        USE_SYSTEM_FP16=ON
#        USE_SYSTEM_PTHREADPOOL=ON
#        USE_SYSTEM_PSIMD=ON
#        USE_SYSTEM_FXDIV=ON
#        USE_SYSTEM_BENCHMARK=ON
#        USE_SYSTEM_ONNX=ON
#        USE_SYSTEM_XNNPACK=ON
#        USE_SYSTEM_PYBIND11=ON
#        USE_SYSTEM_NCCL=ON
#        USE_SYSTEM_NVTX=ON
#
#   USE_MIMALLOC
#      Static link mimalloc into C10, and use mimalloc in alloc_cpu & alloc_free.
#      By default, It is only enabled on Windows.
#
#   USE_PRIORITIZED_TEXT_FOR_LD
#      Uses prioritized text form cmake/prioritized_text.txt for LD
#
#   BUILD_LIBTORCH_WHL
#      Builds libtorch.so and its dependencies as a wheel
#
#   BUILD_PYTHON_ONLY
#      Builds pytorch as a wheel using libtorch.so from a separate wheel

from __future__ import annotations

import os
import sys


if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. "
        "Please switch to 64-bit Python.",
        file=sys.stderr,
    )
    sys.exit(-1)

import platform


# Also update `project.requires-python` in pyproject.toml when changing this
python_min_version = (3, 9, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        f"You are using Python {platform.python_version()}. "
        f"Python >={python_min_version_str} is required.",
        file=sys.stderr,
    )
    sys.exit(-1)

import filecmp
import glob
import importlib
import itertools
import json
import shutil
import subprocess
import sysconfig
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, IO

import setuptools.command.bdist_wheel
import setuptools.command.build_ext
import setuptools.command.sdist
import setuptools.errors
from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution


CWD = Path(__file__).absolute().parent

# Add the current directory to the Python path so that we can import `tools`.
# This is required when running this script with a PEP-517-enabled build backend.
#
# From the PEP-517 documentation: https://peps.python.org/pep-0517
#
# > When importing the module path, we do *not* look in the directory containing
# > the source tree, unless that would be on `sys.path` anyway (e.g. because it
# > is specified in `PYTHONPATH`).
#
sys.path.insert(0, str(CWD))  # this only affects the current process
# Add the current directory to PYTHONPATH so that we can import `tools` in subprocesses
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        str(CWD),
        os.getenv("PYTHONPATH", ""),
    ]
).rstrip(os.pathsep)

from tools.build_pytorch_libs import build_pytorch
from tools.generate_torch_version import get_torch_version
from tools.setup_helpers.cmake import CMake, CMakeValue
from tools.setup_helpers.env import (
    BUILD_DIR,
    build_type,
    IS_DARWIN,
    IS_LINUX,
    IS_WINDOWS,
)
from tools.setup_helpers.generate_linker_script import gen_linker_script


def str2bool(value: str | None) -> bool:
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")


def _get_package_path(package_name: str) -> Path:
    from importlib.util import find_spec

    spec = find_spec(package_name)
    if spec:
        # The package might be a namespace package, so get_data may fail
        try:
            loader = spec.loader
            if loader is not None:
                file_path = loader.get_filename()  # type: ignore[attr-defined]
                return Path(file_path).parent
        except AttributeError:
            pass
    return CWD / package_name


BUILD_LIBTORCH_WHL = str2bool(os.getenv("BUILD_LIBTORCH_WHL"))
BUILD_PYTHON_ONLY = str2bool(os.getenv("BUILD_PYTHON_ONLY"))

# set up appropriate env variables
if BUILD_LIBTORCH_WHL:
    # Set up environment variables for ONLY building libtorch.so and not libtorch_python.so
    # functorch is not supported without python
    os.environ["BUILD_FUNCTORCH"] = "OFF"

if BUILD_PYTHON_ONLY:
    os.environ["BUILD_LIBTORCHLESS"] = "ON"
    os.environ["LIBTORCH_LIB_PATH"] = (_get_package_path("torch") / "lib").as_posix()

################################################################################
# Parameters parsed from environment
################################################################################

VERBOSE_SCRIPT = str2bool(os.getenv("VERBOSE", "1"))
RUN_BUILD_DEPS = True
# see if the user passed a quiet flag to setup.py arguments and respect
# that in our parts of the build
EMIT_BUILD_WARNING = False
RERUN_CMAKE = str2bool(os.environ.pop("CMAKE_FRESH", None))
CMAKE_ONLY = str2bool(os.environ.pop("CMAKE_ONLY", None))
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--cmake":
        RERUN_CMAKE = True
        continue
    if arg == "--cmake-only":
        # Stop once cmake terminates. Leave users a chance to adjust build
        # options.
        CMAKE_ONLY = True
        continue
    if arg == "rebuild" or arg == "build":
        arg = "build"  # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False
    if arg in ["clean", "dist_info", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

if VERBOSE_SCRIPT:

    def report(
        *args: Any, file: IO[str] = sys.stderr, flush: bool = True, **kwargs: Any
    ) -> None:
        print(*args, file=file, flush=flush, **kwargs)

else:

    def report(
        *args: Any, file: IO[str] = sys.stderr, flush: bool = True, **kwargs: Any
    ) -> None:
        pass

    # Make distutils respect --quiet too
    setuptools.distutils.log.warn = report  # type: ignore[attr-defined]

# Constant known variables used throughout this file
TORCH_DIR = CWD / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
THIRD_PARTY_DIR = CWD / "third_party"

# CMAKE: full path to python library
if IS_WINDOWS:
    CMAKE_PYTHON_LIBRARY = (
        Path(sysconfig.get_config_var("prefix"))
        / "libs"
        / f"python{sysconfig.get_config_var('VERSION')}.lib"
    )
    # Fix virtualenv builds
    if not CMAKE_PYTHON_LIBRARY.exists():
        CMAKE_PYTHON_LIBRARY = (
            Path(sys.base_prefix)
            / "libs"
            / f"python{sysconfig.get_config_var('VERSION')}.lib"
        )
else:
    CMAKE_PYTHON_LIBRARY = Path(
        sysconfig.get_config_var("LIBDIR")
    ) / sysconfig.get_config_var("INSTSONAME")


################################################################################
# Version, create_version_file, and package_name
################################################################################

TORCH_PACKAGE_NAME = os.getenv("TORCH_PACKAGE_NAME", "torch")
LIBTORCH_PKG_NAME = os.getenv("LIBTORCH_PACKAGE_NAME", "torch_no_python")
if BUILD_LIBTORCH_WHL:
    TORCH_PACKAGE_NAME = LIBTORCH_PKG_NAME

TORCH_VERSION = get_torch_version()
report(f"Building wheel {TORCH_PACKAGE_NAME}-{TORCH_VERSION}")

cmake = CMake()


def get_submodule_folders() -> list[Path]:
    git_modules_file = CWD / ".gitmodules"
    default_modules_path = [
        THIRD_PARTY_DIR / name
        for name in [
            "gloo",
            "cpuinfo",
            "onnx",
            "fbgemm",
            "cutlass",
        ]
    ]
    if not git_modules_file.exists():
        return default_modules_path
    with git_modules_file.open(encoding="utf-8") as f:
        return [
            CWD / line.partition("=")[-1].strip()
            for line in f
            if line.strip().startswith("path")
        ]


def check_submodules() -> None:
    def check_for_files(folder: Path, files: list[str]) -> None:
        if not any((folder / f).exists() for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder: Path) -> bool:
        return not folder.exists() or (
            folder.is_dir() and next(folder.iterdir(), None) is None
        )

    if str2bool(os.getenv("USE_SYSTEM_LIBS")):
        return
    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            report(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=CWD
            )
            end = time.time()
            report(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            report(" --- Submodule initialization failed")
            report("Please run:\n\tgit submodule update --init --recursive")
            sys.exit(1)
    for folder in folders:
        check_for_files(
            folder,
            [
                "CMakeLists.txt",
                "Makefile",
                "setup.py",
                "LICENSE",
                "LICENSE.md",
                "LICENSE.txt",
            ],
        )
    check_for_files(
        THIRD_PARTY_DIR / "fbgemm" / "external" / "asmjit",
        ["CMakeLists.txt"],
    )


# Windows has very bad support for symbolic links.
# Instead of using symlinks, we're going to copy files over
def mirror_files_into_torchgen() -> None:
    # (new_path, orig_path)
    # Directories are OK and are recursively mirrored.
    paths = [
        (
            CWD / "torchgen/packaged/ATen/native/native_functions.yaml",
            CWD / "aten/src/ATen/native/native_functions.yaml",
        ),
        (
            CWD / "torchgen/packaged/ATen/native/tags.yaml",
            CWD / "aten/src/ATen/native/tags.yaml",
        ),
        (
            CWD / "torchgen/packaged/ATen/templates",
            CWD / "aten/src/ATen/templates",
        ),
        (
            CWD / "torchgen/packaged/autograd",
            CWD / "tools/autograd",
        ),
        (
            CWD / "torchgen/packaged/autograd/templates",
            CWD / "tools/autograd/templates",
        ),
    ]
    for new_path, orig_path in paths:
        # Create the dirs involved in new_path if they don't exist
        if not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the files from the orig location to the new location
        if orig_path.is_file():
            shutil.copyfile(orig_path, new_path)
            continue
        if orig_path.is_dir():
            if new_path.exists():
                # copytree fails if the tree exists already, so remove it.
                shutil.rmtree(new_path)
            shutil.copytree(orig_path, new_path)
            continue
        raise RuntimeError("Check the file paths in `mirror_files_into_torchgen()`")


# all the work we need to do _before_ setup runs
def build_deps() -> None:
    report(f"-- Building version {TORCH_VERSION}")
    check_submodules()
    check_pydep("yaml", "pyyaml")
    build_pytorch(
        version=TORCH_VERSION,
        cmake_python_library=CMAKE_PYTHON_LIBRARY.as_posix(),
        build_python=not BUILD_LIBTORCH_WHL,
        rerun_cmake=RERUN_CMAKE,
        cmake_only=CMAKE_ONLY,
        cmake=cmake,
    )

    if CMAKE_ONLY:
        report(
            'Finished running cmake. Run "ccmake build" or '
            '"cmake-gui build" to adjust build options and '
            '"python -m pip install --no-build-isolation -v ." to build.'
        )
        sys.exit()

    # Use copies instead of symbolic files.
    # Windows has very poor support for them.
    sym_files = [
        CWD / "tools/shared/_utils_internal.py",
        CWD / "torch/utils/benchmark/utils/valgrind_wrapper/callgrind.h",
        CWD / "torch/utils/benchmark/utils/valgrind_wrapper/valgrind.h",
    ]
    orig_files = [
        CWD / "torch/_utils_internal.py",
        CWD / "third_party/valgrind-headers/callgrind.h",
        CWD / "third_party/valgrind-headers/valgrind.h",
    ]
    for sym_file, orig_file in zip(sym_files, orig_files):
        same = False
        if sym_file.exists():
            if filecmp.cmp(sym_file, orig_file):
                same = True
            else:
                sym_file.unlink()
        if not same:
            shutil.copyfile(orig_file, sym_file)


################################################################################
# Building dependent libraries
################################################################################

missing_pydep = """
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
""".strip()


def check_pydep(importname: str, module: str) -> None:
    try:
        importlib.import_module(importname)
    except ImportError as e:
        raise RuntimeError(
            missing_pydep.format(importname=importname, module=module)
        ) from e


class build_ext(setuptools.command.build_ext.build_ext):
    def _embed_libomp(self) -> None:
        # Copy libiomp5.dylib/libomp.dylib inside the wheel package on MacOS
        build_lib = Path(self.build_lib)
        build_torch_lib_dir = build_lib / "torch" / "lib"
        build_torch_include_dir = build_lib / "torch" / "include"
        libtorch_cpu_path = build_torch_lib_dir / "libtorch_cpu.dylib"
        if not libtorch_cpu_path.exists():
            return
        # Parse libtorch_cpu load commands
        otool_cmds = (
            subprocess.check_output(["otool", "-l", str(libtorch_cpu_path)])
            .decode("utf-8")
            .split("\n")
        )
        rpaths: list[str] = []
        libs: list[str] = []
        for idx, line in enumerate(otool_cmds):
            if line.strip() == "cmd LC_LOAD_DYLIB":
                lib_name = otool_cmds[idx + 2].strip()
                assert lib_name.startswith("name ")
                libs.append(lib_name.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

            if line.strip() == "cmd LC_RPATH":
                rpath = otool_cmds[idx + 2].strip()
                assert rpath.startswith("path ")
                rpaths.append(rpath.split(" ", 1)[1].rsplit("(", 1)[0][:-1])

        omplib_path: str = get_cmake_cache_vars()["OpenMP_libomp_LIBRARY"]  # type: ignore[assignment]
        omplib_name: str = get_cmake_cache_vars()["OpenMP_C_LIB_NAMES"]  # type: ignore[assignment]
        omplib_name += ".dylib"
        omplib_rpath_path = os.path.join("@rpath", omplib_name)

        # This logic is fragile and checks only two cases:
        # - libtorch_cpu depends on `@rpath/libomp.dylib`e (happens when built inside miniconda environment)
        # - libtorch_cpu depends on `/abs/path/to/libomp.dylib` (happens when built with libomp from homebrew)
        if not any(c in libs for c in [omplib_path, omplib_rpath_path]):
            return

        # Copy libomp/libiomp5 from rpath locations
        target_lib = build_torch_lib_dir / omplib_name
        libomp_relocated = False
        install_name_tool_args: list[str] = []
        for rpath in rpaths:
            source_lib = os.path.join(rpath, omplib_name)
            if not os.path.exists(source_lib):
                continue
            self.copy_file(source_lib, target_lib)
            # Delete old rpath and add @loader_lib to the rpath
            # This should prevent delocate from attempting to package another instance
            # of OpenMP library in torch wheel as well as loading two libomp.dylib into
            # the address space, as libraries are cached by their unresolved names
            install_name_tool_args = [
                "-rpath",
                rpath,
                "@loader_path",
            ]
            libomp_relocated = True
            break
        if not libomp_relocated and os.path.exists(omplib_path):
            self.copy_file(omplib_path, target_lib)
            install_name_tool_args = [
                "-change",
                omplib_path,
                omplib_rpath_path,
            ]
            if "@loader_path" not in rpaths:
                install_name_tool_args += [
                    "-add_rpath",
                    "@loader_path",
                ]
            libomp_relocated = True
        if libomp_relocated:
            install_name_tool_args = [
                "install_name_tool",
                *install_name_tool_args,
                str(libtorch_cpu_path),
            ]
            subprocess.check_call(install_name_tool_args)
        # Copy omp.h from OpenMP_C_FLAGS and copy it into include folder
        omp_cflags: str = get_cmake_cache_vars()["OpenMP_C_FLAGS"]  # type: ignore[assignment]
        if not omp_cflags:
            return
        for include_dir in [
            Path(f.removeprefix("-I"))
            for f in omp_cflags.split(" ")
            if f.startswith("-I")
        ]:
            omp_h = include_dir / "omp.h"
            if not omp_h.exists():
                continue
            target_omp_h = build_torch_include_dir / "omp.h"
            self.copy_file(omp_h, target_omp_h)
            break

    def run(self) -> None:
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists
        # and we can get an accurate report on what is used and what is not.
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())
        if cmake_cache_vars["USE_NUMPY"]:
            report("-- Building with NumPy bindings")
        else:
            report("-- NumPy not found")
        if cmake_cache_vars["USE_CUDNN"]:
            report(
                "-- Detected cuDNN at "
                f"{cmake_cache_vars['CUDNN_LIBRARY']}, "
                f"{cmake_cache_vars['CUDNN_INCLUDE_DIR']}"
            )
        else:
            report("-- Not using cuDNN")
        if cmake_cache_vars["USE_CUDA"]:
            report(f"-- Detected CUDA at {cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR']}")
        else:
            report("-- Not using CUDA")
        if cmake_cache_vars["USE_XPU"]:
            report(f"-- Detected XPU runtime at {cmake_cache_vars['SYCL_LIBRARY_DIR']}")
        else:
            report("-- Not using XPU")
        if cmake_cache_vars["USE_MKLDNN"]:
            report("-- Using MKLDNN")
            if cmake_cache_vars["USE_MKLDNN_ACL"]:
                report("-- Using Compute Library for the Arm architecture with MKLDNN")
            else:
                report(
                    "-- Not using Compute Library for the Arm architecture with MKLDNN"
                )
            if cmake_cache_vars["USE_MKLDNN_CBLAS"]:
                report("-- Using CBLAS in MKLDNN")
            else:
                report("-- Not using CBLAS in MKLDNN")
        else:
            report("-- Not using MKLDNN")
        if cmake_cache_vars["USE_NCCL"] and cmake_cache_vars["USE_SYSTEM_NCCL"]:
            report(
                "-- Using system provided NCCL library at "
                f"{cmake_cache_vars['NCCL_LIBRARIES']}, "
                f"{cmake_cache_vars['NCCL_INCLUDE_DIRS']}"
            )
        elif cmake_cache_vars["USE_NCCL"]:
            report("-- Building NCCL library")
        else:
            report("-- Not using NCCL")
        if cmake_cache_vars["USE_DISTRIBUTED"]:
            if IS_WINDOWS:
                report("-- Building without distributed package")
            else:
                report("-- Building with distributed package: ")
                report(f"  -- USE_TENSORPIPE={cmake_cache_vars['USE_TENSORPIPE']}")
                report(f"  -- USE_GLOO={cmake_cache_vars['USE_GLOO']}")
                report(f"  -- USE_MPI={cmake_cache_vars['USE_OPENMPI']}")
        else:
            report("-- Building without distributed package")
        if cmake_cache_vars["STATIC_DISPATCH_BACKEND"]:
            report(
                "-- Using static dispatch with "
                f"backend {cmake_cache_vars['STATIC_DISPATCH_BACKEND']}"
            )
        if cmake_cache_vars["USE_LIGHTWEIGHT_DISPATCH"]:
            report("-- Using lightweight dispatch")

        if cmake_cache_vars["USE_ITT"]:
            report("-- Using ITT")
        else:
            report("-- Not using ITT")

        # Do not use clang to compile extensions if `-fstack-clash-protection` is defined
        # in system CFLAGS
        c_flags = os.getenv("CFLAGS", "")
        if (
            IS_LINUX
            and "-fstack-clash-protection" in c_flags
            and "clang" in os.getenv("CC", "")
        ):
            os.environ["CC"] = str(os.environ["CC"])

        super().run()

        if IS_DARWIN:
            self._embed_libomp()

        # Copy the essential export library to compile C++ extensions.
        if IS_WINDOWS:
            build_temp = Path(self.build_temp)
            build_lib = Path(self.build_lib)

            ext_filename = self.get_ext_filename("_C")
            lib_filename = ".".join(ext_filename.split(".")[:-1]) + ".lib"

            export_lib = build_temp / "torch" / "csrc" / lib_filename
            target_lib = build_lib / "torch" / "lib" / "_C.lib"

            # Create "torch/lib" directory if not exists.
            # (It is not created yet in "develop" mode.)
            target_dir = target_lib.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            self.copy_file(export_lib, target_lib)

            # In ROCm on Windows case copy rocblas and hipblaslt files into
            # torch/lib/rocblas/library and torch/lib/hipblaslt/library
            if str2bool(os.getenv("USE_ROCM")):
                rocm_dir_path = Path(os.environ["ROCM_DIR"])
                rocm_bin_path = rocm_dir_path / "bin"
                rocblas_dir = rocm_bin_path / "rocblas"
                target_rocblas_dir = target_dir / "rocblas"
                target_rocblas_dir.mkdir(parents=True, exist_ok=True)
                self.copy_tree(rocblas_dir, str(target_rocblas_dir))

                hipblaslt_dir = rocm_bin_path / "hipblaslt"
                target_hipblaslt_dir = target_dir / "hipblaslt"
                target_hipblaslt_dir.mkdir(parents=True, exist_ok=True)
                self.copy_tree(hipblaslt_dir, str(target_hipblaslt_dir))
            else:
                report("The specified environment variable does not exist.")

    def build_extensions(self) -> None:
        self.create_compile_commands()

        build_lib = Path(self.build_lib).resolve()

        # Copy functorch extension
        for ext in self.extensions:
            if ext.name != "functorch._C":
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = Path(self.get_ext_filename(fullname))
            src = filename.with_stem("functorch")
            dst = build_lib / filename
            if src.exists():
                report(f"Copying {ext.name} from {src} to {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(src, dst)

        super().build_extensions()

    def get_outputs(self) -> list[str]:
        outputs = super().get_outputs()
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report(f"setup.py::get_outputs returning {outputs}")
        return outputs

    def create_compile_commands(self) -> None:
        def load(file: Path) -> list[dict[str, Any]]:
            return json.loads(file.read_text(encoding="utf-8"))

        ninja_files = (CWD / BUILD_DIR).glob("*compile_commands.json")
        cmake_files = (CWD / "torch" / "lib" / "build").glob("*/compile_commands.json")
        all_commands = [
            entry
            for f in itertools.chain(ninja_files, cmake_files)
            for entry in load(f)
        ]

        # cquery does not like c++ compiles that start with gcc.
        # It forgets to include the c++ header directories.
        # We can work around this by replacing the gcc calls that python
        # setup.py generates with g++ calls instead
        for command in all_commands:
            if command["command"].startswith("gcc "):
                command["command"] = "g++ " + command["command"][4:]

        new_contents = json.dumps(all_commands, indent=2)
        contents = ""
        compile_commands_json = CWD / "compile_commands.json"
        if compile_commands_json.exists():
            contents = compile_commands_json.read_text(encoding="utf-8")
        if contents != new_contents:
            compile_commands_json.write_text(new_contents, encoding="utf-8")


class concat_license_files:
    """Merge LICENSE and LICENSES_BUNDLED.txt as a context manager

    LICENSE is the main PyTorch license, LICENSES_BUNDLED.txt is auto-generated
    from all the licenses found in ./third_party/. We concatenate them so there
    is a single license file in the sdist and wheels with all of the necessary
    licensing info.
    """

    def __init__(self, include_files: bool = False) -> None:
        self.f1 = CWD / "LICENSE"
        self.f2 = THIRD_PARTY_DIR / "LICENSES_BUNDLED.txt"
        self.include_files = include_files
        self.bsd_text = ""

    def __enter__(self) -> None:
        """Concatenate files"""

        old_path = sys.path
        sys.path.append(str(THIRD_PARTY_DIR))
        try:
            from build_bundled import create_bundled  # type: ignore[import-not-found]
        finally:
            sys.path = old_path

        self.bsd_text = self.f1.read_text(encoding="utf-8")

        with self.f1.open(mode="a", encoding="utf-8") as f1:
            f1.write("\n\n")
            create_bundled(
                str(THIRD_PARTY_DIR.resolve()),
                f1,
                include_files=self.include_files,
            )

    def __exit__(self, *exc_info: object) -> None:
        """Restore content of f1"""
        self.f1.write_text(self.bsd_text, encoding="utf-8")


# Need to create the proper LICENSE.txt for the wheel
class bdist_wheel(setuptools.command.bdist_wheel.bdist_wheel):
    def run(self) -> None:
        with concat_license_files(include_files=True):
            super().run()

    def write_wheelfile(self, *args: Any, **kwargs: Any) -> None:
        super().write_wheelfile(*args, **kwargs)

        if BUILD_LIBTORCH_WHL:
            assert self.bdist_dir is not None
            bdist_dir = Path(self.bdist_dir)
            # Remove extraneneous files in the libtorch wheel
            for file in itertools.chain(
                bdist_dir.rglob("*.a"),
                bdist_dir.rglob("*.so"),
            ):
                if (bdist_dir / file.name).is_file():
                    file.unlink()
            for file in bdist_dir.rglob("*.py"):
                file.unlink()
            # need an __init__.py file otherwise we wouldn't have a package
            (bdist_dir / "torch" / "__init__.py").touch()


class clean(Command):
    user_options: ClassVar[list[tuple[str, str | None, str]]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        ignores = (CWD / ".gitignore").read_text(encoding="utf-8")
        for wildcard in filter(None, ignores.splitlines()):
            if wildcard.strip().startswith("#"):
                if "BEGIN NOT-CLEAN-FILES" in wildcard:
                    # Marker is found and stop reading .gitignore.
                    break
                # Ignore lines which begin with '#'.
            else:
                # Don't remove absolute paths from the system
                wildcard = wildcard.lstrip("./")
                for filename in glob.iglob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)


# Need to dump submodule hashes and create the proper LICENSE.txt for the sdist
class sdist(setuptools.command.sdist.sdist):
    def run(self) -> None:
        with concat_license_files():
            super().run()


def get_cmake_cache_vars() -> defaultdict[str, CMakeValue]:
    try:
        return defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist.
        # Probably running "python setup.py clean" over a clean directory.
        return defaultdict(lambda: False)


def configure_extension_build() -> tuple[
    list[Extension],  # ext_modules
    dict[str, type[Command]],  # cmdclass
    list[str],  # packages
    dict[str, list[str]],  # entry_points
    list[str],  # extra_install_requires
]:
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """

    cmake_cache_vars = get_cmake_cache_vars()

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs: list[str] = [str(TORCH_LIB_DIR)]
    extra_install_requires: list[str] = []

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

    main_compile_args: list[str] = []
    main_libraries: list[str] = ["torch_python"]

    main_link_args: list[str] = []
    main_sources: list[str] = ["torch/csrc/stub.c"]

    if BUILD_LIBTORCH_WHL:
        main_libraries = ["torch"]
        main_sources = []

    if build_type.is_debug():
        if IS_WINDOWS:
            extra_compile_args += ["/Z7"]
            extra_link_args += ["/DEBUG:FULL"]
        else:
            extra_compile_args += ["-O0", "-g"]
            extra_link_args += ["-O0", "-g"]

    if build_type.is_rel_with_deb_info():
        if IS_WINDOWS:
            extra_compile_args += ["/Z7"]
            extra_link_args += ["/DEBUG:FULL"]
        else:
            extra_compile_args += ["-g"]
            extra_link_args += ["-g"]

    # pypi cuda package that requires installation of cuda runtime, cudnn and cublas
    # should be included in all wheels uploaded to pypi
    pytorch_extra_install_requires = os.getenv("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
    if pytorch_extra_install_requires:
        report(f"pytorch_extra_install_requirements: {pytorch_extra_install_requires}")
        extra_install_requires.extend(
            map(str.strip, pytorch_extra_install_requires.split("|"))
        )

    # Cross-compile for M1
    if IS_DARWIN:
        macos_target_arch = os.getenv("CMAKE_OSX_ARCHITECTURES", "")
        if macos_target_arch in ["arm64", "x86_64"]:
            macos_sysroot_path = os.getenv("CMAKE_OSX_SYSROOT")
            if macos_sysroot_path is None:
                macos_sysroot_path = (
                    subprocess.check_output(
                        ["xcrun", "--show-sdk-path", "--sdk", "macosx"]
                    )
                    .decode("utf-8")
                    .strip()
                )
            extra_compile_args += [
                "-arch",
                macos_target_arch,
                "-isysroot",
                macos_sysroot_path,
            ]
            extra_link_args += ["-arch", macos_target_arch]

    def make_relative_rpath_args(path: str) -> list[str]:
        if IS_DARWIN:
            return ["-Wl,-rpath,@loader_path/" + path]
        elif IS_WINDOWS:
            return []
        else:
            return ["-Wl,-rpath,$ORIGIN/" + path]

    ################################################################################
    # Declare extensions and package
    ################################################################################

    ext_modules: list[Extension] = []
    # packages that we want to install into site-packages and include them in wheels
    includes = ["torch", "torch.*", "torchgen", "torchgen.*"]
    # exclude folders that they look like Python packages but are not wanted in wheels
    excludes = ["tools", "tools.*", "caffe2", "caffe2.*"]
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        includes.extend(["functorch", "functorch.*"])
    else:
        excludes.extend(["functorch", "functorch.*"])
    packages = find_packages(include=includes, exclude=excludes)
    C = Extension(
        "torch._C",
        libraries=main_libraries,
        sources=main_sources,
        language="c",
        extra_compile_args=[
            *main_compile_args,
            *extra_compile_args,
        ],
        include_dirs=[],
        library_dirs=library_dirs,
        extra_link_args=[
            *extra_link_args,
            *main_link_args,
            *make_relative_rpath_args("lib"),
        ],
    )
    ext_modules.append(C)

    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementation
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        ext_modules.append(Extension(name="functorch._C", sources=[]))

    cmdclass = {
        "bdist_wheel": bdist_wheel,
        "build_ext": build_ext,
        "clean": clean,
        "sdist": sdist,
    }

    entry_points = {
        "console_scripts": [
            "torchrun = torch.distributed.run:main",
        ],
        "torchrun.logs_specs": [
            "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
        ],
    }

    if cmake_cache_vars["USE_DISTRIBUTED"]:
        # Only enable fr_trace command if distributed is enabled
        entry_points["console_scripts"].append(
            "torchfrtrace = tools.flight_recorder.fr_trace:main",
        )
    return ext_modules, cmdclass, packages, entry_points, extra_install_requires


# post run, warnings, printed at the end to make them more visible
build_update_message = """
It is no longer necessary to use the 'build' or 'rebuild' targets

To install:
  $ python -m pip install --no-build-isolation -v .
To develop locally:
  $ python -m pip install --no-build-isolation -v -e .
To force cmake to re-generate native build files (off by default):
  $ CMAKE_FRESH=1 python -m pip install --no-build-isolation -v -e .
""".strip()


def print_box(msg: str) -> None:
    msg = textwrap.dedent(msg).strip()
    lines = ["", *msg.split("\n"), ""]
    max_width = max(len(l) for l in lines)
    print("+" + "-" * (max_width + 4) + "+", file=sys.stderr, flush=True)
    for line in lines:
        print(f"|  {line:<{max_width}s}  |", file=sys.stderr, flush=True)
    print("+" + "-" * (max_width + 4) + "+", file=sys.stderr, flush=True)


def main() -> None:
    if BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY:
        raise RuntimeError(
            "Conflict: 'BUILD_LIBTORCH_WHL' and 'BUILD_PYTHON_ONLY' can't both be 1. "
            "Set one to 0 and rerun."
        )

    install_requires = [
        "filelock",
        "typing-extensions>=4.10.0",
        'setuptools ; python_version >= "3.12"',
        "sympy>=1.13.3",
        "networkx>=2.5.1",
        "jinja2",
        "fsspec>=0.8.5",
    ]
    if BUILD_PYTHON_ONLY:
        install_requires += [f"{LIBTORCH_PKG_NAME}=={TORCH_VERSION}"]

    if str2bool(os.getenv("USE_PRIORITIZED_TEXT_FOR_LD")):
        gen_linker_script(
            filein="cmake/prioritized_text.txt", fout="cmake/linker_script.ld"
        )
        linker_script_path = os.path.abspath("cmake/linker_script.ld")
        os.environ["LDFLAGS"] = os.getenv("LDFLAGS", "") + f" -T{linker_script_path}"
        os.environ["CFLAGS"] = (
            os.getenv("CFLAGS", "") + " -ffunction-sections -fdata-sections"
        )
        os.environ["CXXFLAGS"] = (
            os.getenv("CXXFLAGS", "") + " -ffunction-sections -fdata-sections"
        )
    elif platform.system() == "Linux" and platform.processor() == "aarch64":
        print_box(
            """
            WARNING: we strongly recommend enabling linker script optimization for ARM + CUDA.
            To do so please export USE_PRIORITIZED_TEXT_FOR_LD=1
            """
        )

    # Parse the command line and check the arguments before we proceed with
    # building deps and setup. We need to set values so `--help` works.
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.errors.BaseError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    mirror_files_into_torchgen()
    if RUN_BUILD_DEPS:
        build_deps()

    (
        ext_modules,
        cmdclass,
        packages,
        entry_points,
        extra_install_requires,
    ) = configure_extension_build()
    install_requires += extra_install_requires

    torch_package_data = [
        "py.typed",
        "bin/*",
        "test/*",
        "*.pyi",
        "**/*.pyi",
        "lib/*.pdb",
        "lib/**/*.pdb",
        "lib/*shm*",
        "lib/torch_shm_manager",
        "lib/*.h",
        "lib/**/*.h",
        "include/*.h",
        "include/**/*.h",
        "include/*.hpp",
        "include/**/*.hpp",
        "include/*.cuh",
        "include/**/*.cuh",
        "csrc/inductor/aoti_runtime/model.h",
        "_inductor/codegen/*.h",
        "_inductor/codegen/aoti_runtime/*.h",
        "_inductor/codegen/aoti_runtime/*.cpp",
        "_inductor/script.ld",
        "_export/serde/*.yaml",
        "_export/serde/*.thrift",
        "share/cmake/ATen/*.cmake",
        "share/cmake/Caffe2/*.cmake",
        "share/cmake/Caffe2/public/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake",
        "share/cmake/Gloo/*.cmake",
        "share/cmake/Tensorpipe/*.cmake",
        "share/cmake/Torch/*.cmake",
        "utils/benchmark/utils/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.h",
        "utils/model_dump/skeleton.html",
        "utils/model_dump/code.js",
        "utils/model_dump/*.mjs",
        "_dynamo/graph_break_registry.json",
        "tools/dynamo/gb_id_mapping.py",
    ]

    if not BUILD_LIBTORCH_WHL:
        torch_package_data += [
            "lib/libtorch_python.so",
            "lib/libtorch_python.dylib",
            "lib/libtorch_python.dll",
        ]
    if not BUILD_PYTHON_ONLY:
        torch_package_data += [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
        # XXX: Why not use wildcards ["lib/aotriton.images/*", "lib/aotriton.images/**/*"] here?
        aotriton_image_path = TORCH_DIR / "lib" / "aotriton.images"
        aks2_files = [
            file.relative_to(TORCH_DIR).as_posix()
            for file in aotriton_image_path.rglob("*")
            if file.is_file()
        ]
        torch_package_data += aks2_files
    if get_cmake_cache_vars()["USE_TENSORPIPE"]:
        torch_package_data += [
            "include/tensorpipe/*.h",
            "include/tensorpipe/**/*.h",
        ]
    if get_cmake_cache_vars()["USE_KINETO"]:
        torch_package_data += [
            "include/kineto/*.h",
            "include/kineto/**/*.h",
        ]
    torchgen_package_data = [
        "packaged/*",
        "packaged/**/*",
    ]
    package_data = {
        "torch": torch_package_data,
    }
    exclude_package_data = {}

    if not BUILD_LIBTORCH_WHL:
        package_data["torchgen"] = torchgen_package_data
        exclude_package_data["torchgen"] = ["*.py[co]"]
    else:
        # no extensions in BUILD_LIBTORCH_WHL mode
        ext_modules = []

    setup(
        name=TORCH_PACKAGE_NAME,
        version=TORCH_VERSION,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=install_requires,
        package_data=package_data,
        exclude_package_data=exclude_package_data,
        # Disable automatic inclusion of data files because we want to
        # explicitly control with `package_data` above.
        include_package_data=False,
    )
    if EMIT_BUILD_WARNING:
        print_box(build_update_message)


if __name__ == "__main__":
    main()
