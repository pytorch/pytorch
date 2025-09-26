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
#   USE_FBGEMM_GENAI=0
#     disables the FBGEMM GenAI build
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
#   USE_ROCM_CK_GEMM=1
#     Enable building CK GEMM backend in ROCm platform
#
#   USE_ROCM_CK_SDPA=1
#     Enable building CK SDPA backend in ROCm platform
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
#   BUILD_LIBTORCH_WHL
#      Builds libtorch.so and its dependencies as a wheel
#
#   BUILD_PYTHON_ONLY
#      Builds pytorch as a wheel using libtorch.so from a separate wheel
#
#   USE_NIGHTLY=VERSION
#      Skip cmake build and instead download and extract nightly PyTorch wheel
#      matching the specified version (e.g., USE_NIGHTLY="2.8.0.dev20250608+cpu")
#      into the local directory for development use

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
python_min_version = (3, 10, 0)
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
import tempfile
import textwrap
import time
import zipfile
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
    if arg == "develop":
        print(
            (
                "WARNING: Redirecting 'python setup.py develop' to 'pip install -e . -v --no-build-isolation',"
                " for more info see https://github.com/pytorch/pytorch/issues/152276"
            ),
            file=sys.stderr,
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "-v",
                "--no-build-isolation",
            ],
            env={**os.environ},
        )
        sys.exit(result.returncode)
    if arg == "install":
        print(
            (
                "WARNING: Redirecting 'python setup.py install' to 'pip install . -v --no-build-isolation',"
                " for more info see https://github.com/pytorch/pytorch/issues/152276"
            ),
            file=sys.stderr,
        )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "-v", "--no-build-isolation"],
            env={**os.environ},
        )
        sys.exit(result.returncode)
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


# ATTENTION: THIS IS AI SLOP
def extract_variant_from_version(version: str) -> str:
    """Extract variant from version string, defaulting to 'cpu'."""
    import re

    variant_match = re.search(r"\+([^-\s,)]+)", version)
    return variant_match.group(1) if variant_match else "cpu"


# ATTENTION: THIS IS AI SLOP
def get_nightly_git_hash(version: str) -> str:
    """Download a nightly wheel and extract the git hash from its version.py file."""
    # Extract variant from version to construct correct URL
    variant = extract_variant_from_version(version)
    nightly_index_url = f"https://download.pytorch.org/whl/nightly/{variant}/"

    torch_version_spec = f"torch=={version}"

    # Create a temporary directory for downloading
    with tempfile.TemporaryDirectory(prefix="pytorch-hash-extract-") as temp_dir:
        temp_path = Path(temp_dir)

        # Download the wheel
        report(f"-- Downloading {version} wheel to extract git hash...")
        download_cmd = [
            "uvx",
            "pip",
            "download",
            "--index-url",
            nightly_index_url,
            "--pre",
            "--no-deps",
            "--dest",
            str(temp_path),
            torch_version_spec,
        ]

        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download {version} wheel for git hash extraction: {result.stderr}"
            )

        # Find the downloaded wheel file
        wheel_files = list(temp_path.glob("torch-*.whl"))
        if not wheel_files:
            raise RuntimeError(f"No torch wheel found after downloading {version}")

        wheel_file = wheel_files[0]

        # Extract the wheel and look for version.py
        with tempfile.TemporaryDirectory(
            prefix="pytorch-wheel-extract-"
        ) as extract_dir:
            extract_path = Path(extract_dir)

            with zipfile.ZipFile(wheel_file, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Find torch directory and version.py
            torch_dirs = list(extract_path.glob("torch"))
            if not torch_dirs:
                torch_dirs = list(extract_path.glob("*/torch"))

            if not torch_dirs:
                raise RuntimeError(f"Could not find torch directory in {version} wheel")

            version_file = torch_dirs[0] / "version.py"
            if not version_file.exists():
                raise RuntimeError(f"Could not find version.py in {version} wheel")

            # Read and parse version.py to extract git_version (nightly branch commit)
            from ast import literal_eval

            nightly_commit = None
            with version_file.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("git_version"):
                        try:
                            # Parse the git_version assignment, e.g., git_version = "abc123def456"
                            nightly_commit = literal_eval(
                                line.partition("=")[2].strip()
                            )
                            break
                        except (ValueError, SyntaxError):
                            continue

            if not nightly_commit:
                raise RuntimeError(
                    f"Could not parse git_version from {version} wheel's version.py"
                )

            # Now fetch the nightly branch and extract the real source commit from the message
            report("-- Fetching nightly branch to extract source commit...")

            # Fetch only the nightly branch
            subprocess.check_call(["git", "fetch", "origin", "nightly"], cwd=str(CWD))

            # Get the commit message from the nightly commit
            commit_message = subprocess.check_output(
                ["git", "show", "--no-patch", "--format=%s", nightly_commit],
                cwd=str(CWD),
                text=True,
            ).strip()

            # Parse the commit message to extract the real hash
            # Format: "2025-08-06 nightly release (74a754aae98aabc2aca67e5edb41cc684fae9a82)"
            import re

            hash_match = re.search(r"\(([0-9a-fA-F]{40})\)", commit_message)
            if hash_match:
                real_commit = hash_match.group(1)
                report(f"-- Extracted source commit: {real_commit[:12]}...")
                return real_commit
            else:
                raise RuntimeError(
                    f"Could not parse commit hash from nightly commit message: {commit_message}"
                )


# ATTENTION: THIS IS AI SLOP
def get_latest_nightly_version(variant: str = "cpu") -> str:
    """Get the latest available nightly version using pip to query the PyTorch nightly index."""
    # Get the latest available nightly version for the specified variant
    nightly_index_url = f"https://download.pytorch.org/whl/nightly/{variant}/"

    # Run pip index to get available versions
    output = subprocess.check_output(
        [
            "uvx",
            "pip",
            "index",
            "versions",
            "--index-url",
            nightly_index_url,
            "--pre",
            "torch",
        ],
        text=True,
        timeout=30,
    )

    # Parse the first line to get the latest version
    # Format: "torch (2.9.0.dev20250806)" or "torch (2.9.0.dev20250806+cpu)"
    first_line = output.strip().split("\n")[0]
    if "(" in first_line and ")" in first_line:
        # Extract version from parentheses exactly as reported
        version = first_line.split("(")[1].split(")")[0]
        return version

    raise RuntimeError(f"Could not parse version from pip index output: {first_line}")


# ATTENTION: THIS IS AI SLOP
def download_and_extract_nightly_wheel(version: str) -> None:
    """Download and extract nightly PyTorch wheel for USE_NIGHTLY=VERSION builds."""

    # Extract variant from version (e.g., cpu, cu121, cu118, rocm5.7)
    variant = extract_variant_from_version(version)
    nightly_index_url = f"https://download.pytorch.org/whl/nightly/{variant}/"

    # Construct the full torch version spec
    torch_version_spec = f"torch=={version}"

    # Create a temporary directory for downloading
    with tempfile.TemporaryDirectory(prefix="pytorch-nightly-") as temp_dir:
        temp_path = Path(temp_dir)

        # Use pip to download the specific nightly wheel
        download_cmd = [
            "uvx",
            "pip",
            "download",
            "--index-url",
            nightly_index_url,
            "--pre",
            "--no-deps",
            "--dest",
            str(temp_path),
            torch_version_spec,
        ]

        report("-- Downloading nightly PyTorch wheel...")
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Try to get the latest nightly version for the same variant to help the user
            variant = extract_variant_from_version(version)
            try:
                report(f"-- Detecting latest {variant} nightly version...")
                latest_version = get_latest_nightly_version(variant)
                error_msg = f"Failed to download nightly wheel for version {version}: {result.stderr.strip()}"
                error_msg += (
                    f"\n\nLatest available {variant} nightly version: {latest_version}"
                )
                error_msg += f'\nTry: USE_NIGHTLY="{latest_version}"'

                # Also get the git hash for the latest version
                git_hash = get_nightly_git_hash(latest_version)
                error_msg += f"\n\nIMPORTANT: You must checkout the matching source commit:\ngit checkout {git_hash}"
            except Exception:
                # If we can't get latest for this variant, try CPU as fallback
                try:
                    report("-- Detecting latest CPU nightly version...")
                    latest_version = get_latest_nightly_version("cpu")
                    error_msg = f"Failed to download nightly wheel for version {version}: {result.stderr.strip()}"
                    error_msg += f"\n\nCould not find {variant} nightlies. Latest available CPU nightly version: {latest_version}"
                    error_msg += f'\nTry: USE_NIGHTLY="{latest_version}"'
                except Exception:
                    error_msg = f"Failed to download nightly wheel for version {version}: {result.stderr.strip()}"
                    error_msg += "\n\nCould not determine latest nightly version. "
                    error_msg += "Check https://download.pytorch.org/whl/nightly/ for available versions."

            raise RuntimeError(error_msg)

        # Find the downloaded wheel file
        wheel_files = list(temp_path.glob("torch-*.whl"))
        if not wheel_files:
            raise RuntimeError("No torch wheel found after download")
        elif len(wheel_files) > 1:
            raise RuntimeError(f"Multiple torch wheels found: {wheel_files}")

        wheel_file = wheel_files[0]
        report(f"-- Downloaded wheel: {wheel_file.name}")

        # Extract the wheel
        with tempfile.TemporaryDirectory(
            prefix="pytorch-wheel-extract-"
        ) as extract_dir:
            extract_path = Path(extract_dir)

            # Use Python's zipfile to extract the wheel
            with zipfile.ZipFile(wheel_file, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Find the torch directory in the extracted wheel
            torch_dirs = list(extract_path.glob("torch"))
            if not torch_dirs:
                # Sometimes the torch directory might be nested
                torch_dirs = list(extract_path.glob("*/torch"))

            if not torch_dirs:
                raise RuntimeError("Could not find torch directory in extracted wheel")

            source_torch_dir = torch_dirs[0]
            target_torch_dir = TORCH_DIR

            report(
                f"-- Extracting wheel contents from {source_torch_dir} to {target_torch_dir}"
            )

            # Copy the essential files from the wheel to our local directory
            # Based on the file listing logic from tools/nightly.py
            files_to_copy: list[Path] = []

            # Get platform-specific binary files
            if IS_LINUX:
                files_to_copy.extend(source_torch_dir.glob("*.so"))
                files_to_copy.extend(
                    (source_torch_dir / "lib").glob("*.so*")
                    if (source_torch_dir / "lib").exists()
                    else []
                )
            elif IS_DARWIN:
                files_to_copy.extend(source_torch_dir.glob("*.so"))
                files_to_copy.extend(
                    (source_torch_dir / "lib").glob("*.dylib")
                    if (source_torch_dir / "lib").exists()
                    else []
                )
            elif IS_WINDOWS:
                files_to_copy.extend(source_torch_dir.glob("*.pyd"))
                files_to_copy.extend(
                    (source_torch_dir / "lib").glob("*.lib")
                    if (source_torch_dir / "lib").exists()
                    else []
                )
                files_to_copy.extend(
                    (source_torch_dir / "lib").glob("*.dll")
                    if (source_torch_dir / "lib").exists()
                    else []
                )

            # Add essential directories and files
            essential_items = ["version.py", "bin", "include", "lib"]
            for item_name in essential_items:
                item_path = source_torch_dir / item_name
                if item_path.exists():
                    files_to_copy.append(item_path)

            # Add testing internal generated files
            testing_generated = source_torch_dir / "testing" / "_internal" / "generated"
            if testing_generated.exists():
                files_to_copy.append(testing_generated)

            # Copy all the files and directories
            for src_path in files_to_copy:
                rel_path = src_path.relative_to(source_torch_dir)
                dst_path = target_torch_dir / rel_path

                # Copy files and directories, preserving existing subdirectories
                if src_path.is_dir():
                    # Create destination directory if it doesn't exist
                    dst_path.mkdir(parents=True, exist_ok=True)
                    # Copy individual entries from source directory
                    for src_item in src_path.iterdir():
                        dst_item = dst_path / src_item.name
                        if src_item.is_dir():
                            # Recursively copy subdirectories (this will preserve existing ones)
                            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                        else:
                            # Copy individual files, overwriting existing ones
                            shutil.copy2(src_item, dst_item)
                else:
                    # For files, remove existing and copy new
                    if dst_path.exists():
                        dst_path.unlink()
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

                report(f"   Copied {rel_path}")

    report("-- Nightly wheel extraction completed")


# all the work we need to do _before_ setup runs
def build_deps() -> None:
    report(f"-- Building version {TORCH_VERSION}")

    # ATTENTION: THIS IS AI SLOP
    # Check for USE_NIGHTLY=VERSION to bypass normal build and download nightly wheel
    nightly_version = os.getenv("USE_NIGHTLY")
    if nightly_version is not None:
        import re

        if (
            nightly_version == ""
            or nightly_version == "cpu"
            or re.match(r"^cu\d+$", nightly_version)
            or re.match(r"^rocm\d+\.\d+$", nightly_version)
        ):
            # Empty string or variant-only specification, show error with latest version
            variant = "cpu" if nightly_version == "" else nightly_version
            report(f"-- Detecting latest {variant} nightly version...")
            latest_version = get_latest_nightly_version(variant)
            # Also get the git hash to tell user which commit to checkout
            git_hash = get_nightly_git_hash(latest_version)

            if nightly_version == "":
                error_msg = f"USE_NIGHTLY cannot be empty. Latest available version: {latest_version}\n"
            else:
                error_msg = (
                    "USE_NIGHTLY requires a specific version, not just a variant. "
                    "Latest available {nightly_version} version: {latest_version}\n"
                )

            error_msg += f'Try: USE_NIGHTLY="{latest_version}"'
            error_msg += f"\n\nIMPORTANT: You must checkout the matching source commit for this binary:\ngit checkout {git_hash}"
            raise RuntimeError(error_msg)
        else:
            # Full version specification
            report(
                f"-- USE_NIGHTLY={nightly_version} detected, downloading nightly wheel"
            )
            download_and_extract_nightly_wheel(nightly_version)
            return

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
        cmake_cache_vars = get_cmake_cache_vars()
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

    def build_extensions(self) -> None:
        self.create_compile_commands()

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
        "_inductor/kernel/flex/templates/*.jinja",
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
    # some win libraries are excluded
    # these are statically linked
    exclude_windows_libs = [
        "lib/dnnl.lib",
        "lib/kineto.lib",
        "lib/libprotobuf-lite.lib",
        "lib/libprotobuf.lib",
        "lib/libprotoc.lib",
    ]
    exclude_package_data = {
        "torch": exclude_windows_libs,
    }

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
