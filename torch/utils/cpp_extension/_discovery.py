# mypy: allow-untyped-defs
"""Toolchain discovery and build-system-agnostic helpers.

Contains platform detection, CUDA/ROCm/SYCL HOME discovery, common compiler
flags, ABI compatibility checks, and utilities shared by both the
:mod:`torch.utils.cpp_extension` JIT path and the setuptools adapter. Nothing
here imports setuptools.
"""

import glob
import importlib.metadata
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from typing_extensions import deprecated

import torch
import torch._appdirs
from torch.torch_version import TorchVersion, Version

from .._cpp_extension_versioner import ExtensionVersioner


logger = logging.getLogger(__name__)


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform.startswith("darwin")
IS_LINUX = sys.platform.startswith("linux")
LIB_EXT = ".pyd" if IS_WINDOWS else ".so"
EXEC_EXT = ".exe" if IS_WINDOWS else ""
CLIB_PREFIX = "" if IS_WINDOWS else "lib"
CLIB_EXT = ".dll" if IS_WINDOWS else ".so"
SHARED_FLAG = "/DLL" if IS_WINDOWS else "-shared"

_TORCH_PATH = os.path.dirname(torch.__file__)
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, "lib")


SUBPROCESS_DECODE_ARGS = ("oem",) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

VersionRange = tuple[tuple[int, ...], tuple[int, ...]]
VersionMap = dict[str, VersionRange]
# The following values were taken from the following GitHub gist that
# summarizes the minimum valid major versions of g++/clang++ for each supported
# CUDA version: https://gist.github.com/ax3l/9489132
# Or from include/crt/host_config.h in the CUDA SDK
# The second value is the exclusive(!) upper bound, i.e. min <= version < max
CUDA_GCC_VERSIONS: VersionMap = {
    "11.0": (MINIMUM_GCC_VERSION, (10, 0)),
    "11.1": (MINIMUM_GCC_VERSION, (11, 0)),
    "11.2": (MINIMUM_GCC_VERSION, (11, 0)),
    "11.3": (MINIMUM_GCC_VERSION, (11, 0)),
    "11.4": ((6, 0, 0), (12, 0)),
    "11.5": ((6, 0, 0), (12, 0)),
    "11.6": ((6, 0, 0), (12, 0)),
    "11.7": ((6, 0, 0), (12, 0)),
    "12.0": ((6, 0, 0), (13, 0)),
    "12.1": ((6, 0, 0), (13, 0)),
    "12.2": ((6, 0, 0), (13, 0)),
    "12.3": ((6, 0, 0), (14, 0)),
    "12.4": ((6, 0, 0), (14, 0)),
    "12.5": ((6, 0, 0), (14, 0)),
    "12.6": ((6, 0, 0), (14, 0)),
    "12.7": ((6, 0, 0), (14, 0)),
    "12.8": ((6, 0, 0), (15, 0)),
    "12.9": ((6, 0, 0), (15, 0)),
    "13.0": ((6, 0, 0), (16, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {
    "11.1": (MINIMUM_CLANG_VERSION, (11, 0)),
    "11.2": (MINIMUM_CLANG_VERSION, (12, 0)),
    "11.3": (MINIMUM_CLANG_VERSION, (12, 0)),
    "11.4": (MINIMUM_CLANG_VERSION, (13, 0)),
    "11.5": (MINIMUM_CLANG_VERSION, (13, 0)),
    "11.6": (MINIMUM_CLANG_VERSION, (14, 0)),
    "11.7": (MINIMUM_CLANG_VERSION, (14, 0)),
    "12.0": ((7, 0), (15, 0)),
    "12.1": ((7, 0), (15, 0)),
    "12.2": ((7, 0), (16, 0)),
    "12.3": ((7, 0), (16, 0)),
    "12.4": ((7, 0), (17, 0)),
    "12.5": ((7, 0), (18, 0)),
    "12.6": ((7, 0), (18, 0)),
    "12.7": ((7, 0), (19, 0)),
    "12.8": ((7, 0), (19, 0)),
    "12.9": ((7, 0), (19, 0)),
    "13.0": ((7, 0), (21, 0)),
}


# Taken directly from python stdlib < 3.9
# See https://github.com/pytorch/pytorch/issues/48617
def _nt_quote_args(args: list[str] | None) -> list[str]:
    """Quote command-line arguments for DOS/Windows conventions.

    Just wraps every argument which contains blanks in double quotes, and
    returns a new argument list.
    """
    # Cover None-type
    if not args:
        return []
    return [f'"{arg}"' if " " in arg else arg for arg in args]


def _find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*"
                )
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        logger.warning("No CUDA runtime is found, using CUDA_HOME='%s'", cuda_home)
    return cuda_home


def _find_rocm_home() -> str | None:
    """Find the ROCm install path."""
    # Guess #1
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    if rocm_home is None:
        # Guess #2
        hipcc_path = shutil.which("hipcc")
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(os.path.realpath(hipcc_path)))
            # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == "hip":
                rocm_home = os.path.dirname(rocm_home)
        else:
            # Guess #3
            fallback_path = "/opt/rocm"
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        logger.warning("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
    return rocm_home


def _find_sycl_home() -> str | None:
    sycl_home = None
    icpx_path = shutil.which("icpx")
    # Guess 1: for source code build developer/user, we'll have icpx in PATH,
    # which will tell us the SYCL_HOME location.
    if icpx_path is not None:
        sycl_home = os.path.dirname(os.path.dirname(os.path.realpath(icpx_path)))

    # Guess 2: for users install Pytorch with XPU support, the sycl runtime is
    # inside intel-sycl-rt, which is automatically installed via pip dependency.
    else:
        try:
            files = importlib.metadata.files("intel-sycl-rt") or []
            for f in files:
                if f.name == "libsycl.so":
                    sycl_home = os.path.dirname(Path(f.locate()).parent.resolve())
                    break
        except importlib.metadata.PackageNotFoundError:
            logger.warning(
                "Trying to find SYCL_HOME from intel-sycl-rt package, but it is not installed."
            )
    return sycl_home


def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    if ROCM_HOME is None:
        raise OSError(
            "ROCM_HOME environment variable is not set. "
            "Please set it to your ROCm install root."
        )
    return os.path.join(ROCM_HOME, *paths)


def _join_sycl_home(*paths) -> str:
    """
    Join paths with SYCL_HOME, or raises an error if it SYCL_HOME is not found.

    This is basically a lazy way of raising an error for missing SYCL_HOME
    only once we need to get any SYCL-specific path.
    """
    if SYCL_HOME is None:
        raise OSError(
            "SYCL runtime is not dected. Please setup the pytorch "
            "prerequisites for Intel GPU following the instruction in "
            "https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support "
            "or install intel-sycl-rt via pip."
        )

    return os.path.join(SYCL_HOME, *paths)


ABI_INCOMPATIBILITY_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) may be ABI-incompatible with PyTorch!"
    "Please use a compiler that is ABI-compatible with GCC 5.0 and above."
    "See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html."
    "See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6"
    "for instructions on how to install GCC 5 or higher."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
WRONG_COMPILER_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) is not compatible with the compiler Pytorch was"
    "built with for this platform, which is %s on %s. Please"
    "use %s to compile your extension. Alternatively, you may"
    "compile PyTorch from source using %s, and then you can also use"
    "%s to compile your extension."
    "See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help"
    "with compiling PyTorch from source."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
CUDA_MISMATCH_MESSAGE = (
    "The detected CUDA version (%s) mismatches the version that was used to compile"
    "PyTorch (%s). Please make sure to use the same CUDA versions."
)
CUDA_MISMATCH_WARN = "The detected CUDA version (%s) has a minor version mismatch with the version that was used to compile PyTorch (%s). Most likely this shouldn't be a problem."
CUDA_NOT_FOUND_MESSAGE = (
    "CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH"
    "environment variable or add NVCC to your system PATH. The extension compilation will fail."
)
ROCM_HOME = (
    _find_rocm_home() if (torch.cuda._is_compiled() and torch.version.hip) else None
)
HIP_HOME = _join_rocm_home("hip") if ROCM_HOME else None
IS_HIP_EXTENSION = bool(ROCM_HOME is not None and torch.version.hip is not None)
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split(".")[:2])

CUDA_HOME = (
    _find_cuda_home() if (torch.cuda._is_compiled() and torch.version.cuda) else None
)
CUDNN_HOME = os.environ.get("CUDNN_HOME") or os.environ.get("CUDNN_PATH")
SYCL_HOME = _find_sycl_home() if torch.xpu._is_compiled() else None
WINDOWS_CUDA_HOME = os.environ.get(
    "WINDOWS_CUDA_HOME"
)  # used for AOTI cross-compilation

# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+\w+\+\w+")

COMMON_MSVC_FLAGS = [
    "/MD",
    "/wd4819",
    "/wd4251",
    "/wd4244",
    "/wd4267",
    "/wd4275",
    "/wd4018",
    "/wd4190",
    "/wd4624",
    "/wd4067",
    "/wd4068",
    "/EHsc",
]

MSVC_IGNORE_CUDAFE_WARNINGS = [
    "base_class_has_different_dll_interface",
    "field_without_dll_interface",
    "dll_interface_conflict_none_assumed",
    "dll_interface_conflict_dllexport_assumed",
]

COMMON_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
]

COMMON_HIP_FLAGS = [
    "-D__HIP_PLATFORM_AMD__=1",
    "-DUSE_ROCM=1",
    "-DHIPBLAS_V2",
]

if not IS_WINDOWS:
    COMMON_HIP_FLAGS.append("-fPIC")

COMMON_HIPCC_FLAGS = [
    "-DCUDA_HAS_FP16=1",
    "-D__HIP_NO_HALF_OPERATORS__=1",
    "-D__HIP_NO_HALF_CONVERSIONS__=1",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
]

if IS_WINDOWS:
    # Compatibility flags, similar to those set in cmake/Dependencies.cmake.
    COMMON_HIPCC_FLAGS.append("-fms-extensions")
    # Suppress warnings about dllexport.
    COMMON_HIPCC_FLAGS.append("-Wno-ignored-attributes")


def _get_icpx_version() -> str:
    icpx = "icx" if IS_WINDOWS else "icpx"
    compiler_info = subprocess.check_output([icpx, "--version"])
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", compiler_info.decode().strip())
    version = ["0", "0", "0"] if match is None else list(match.groups())
    version = list(map(int, version))
    if len(version) != 3:
        raise AssertionError("Failed to parse DPC++ compiler version")
    # Aligning version format with what torch.version.xpu() returns
    return f"{version[0]}{version[1]:02}{version[2]:02}"


def _get_sycl_arch_list():
    if "TORCH_XPU_ARCH_LIST" in os.environ:
        return os.environ.get("TORCH_XPU_ARCH_LIST")
    arch_list = torch.xpu.get_arch_list()
    # Dropping dg2* archs since they lack hardware support for fp64 and require
    # special consideration from the user. If needed these platforms can
    # be requested thru TORCH_XPU_ARCH_LIST environment variable.
    arch_list = [x for x in arch_list if not x.startswith("dg2")]
    return ",".join(arch_list)


# If arch list returned by _get_sycl_arch_list() is empty, then sycl kernels will be compiled
# for default spir64 target and avoid device specific compilations entirely. Further, kernels
# will be JIT compiled at runtime.
def _append_sycl_targets_if_missing(cflags) -> None:
    if any(flag.startswith("-fsycl-targets=") for flag in cflags):
        # do nothing: user has manually specified sycl targets
        return
    if _get_sycl_arch_list() != "":
        # AOT (spir64_gen) + JIT (spir64)
        cflags.append("-fsycl-targets=spir64_gen,spir64")
    else:
        # JIT (spir64)
        cflags.append("-fsycl-targets=spir64")


def _get_sycl_device_flags(cflags):
    # We need last occurrence of -fsycl-targets as it will be the one taking effect.
    # So searching in reversed list.
    flags = [f for f in reversed(cflags) if f.startswith("-fsycl-targets=")]
    if not flags:
        raise AssertionError("bug: -fsycl-targets should have been amended to cflags")

    arch_list = _get_sycl_arch_list()
    if arch_list != "":
        flags += [f'-Xs "-device {arch_list}"']
    return flags


_COMMON_SYCL_FLAGS = [
    "-fsycl",
]

_SYCL_DLINK_FLAGS = [
    *_COMMON_SYCL_FLAGS,
    "-fsycl-link",
    "--offload-compress",
]

JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    "win32": "x86",
    "win-amd64": "x86_amd64",
}

min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode


def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
    else:
        compiler = os.environ.get("CXX", "c++")
    return compiler


def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> list[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return (
        ["clang++", "clang"]
        if IS_MACOS
        else ["g++", "gcc", "gnu-c++", "gnu-cc", "clang++", "clang"]
    )


def _maybe_write(filename, new_content) -> None:
    r"""
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    """
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, "w") as source_file:
        source_file.write(new_content)


def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname="torch_extensions"))


def check_compiler_ok_for_platform(compiler: str) -> bool:
    """
    Verify that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    if IS_WINDOWS:
        return True
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return False
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(compiler_path)
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except subprocess.CalledProcessError:
        # If '-v' fails, try '--version'
        version_string = subprocess.check_output(
            [compiler, "--version"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            # Though on Ubuntu it's sometimes called "Ubuntu clang version"
            return "clang version" in version_string
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == "c++" and "gcc version" in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def get_compiler_abi_compatibility_and_version(compiler) -> tuple[bool, TorchVersion]:
    """
    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    """
    if not _is_binary_build():
        return (True, TorchVersion("0.0.0"))
    if os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI") in [
        "ON",
        "1",
        "YES",
        "TRUE",
        "Y",
    ]:
        return (True, TorchVersion("0.0.0"))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        logger.warning(
            WRONG_COMPILER_WARNING,
            compiler,
            _accepted_compilers_for_platform()[0],
            sys.platform,
            _accepted_compilers_for_platform()[0],
            compiler,
            compiler,
        )
        return (False, TorchVersion("0.0.0"))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion("0.0.0"))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            compiler_info = subprocess.check_output(
                [compiler, "-dumpfullversion", "-dumpversion"]
            )
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
        match = re.search(
            r"(\d+)\.(\d+)\.(\d+)",
            compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip(),
        )
        version = ["0", "0", "0"] if match is None else list(match.groups())
    except (subprocess.CalledProcessError, OSError):
        logger.warning(
            "Error checking compiler version for %s", compiler, exc_info=True
        )
        return (False, TorchVersion("0.0.0"))

    # convert alphanumeric string to numeric string
    # amdclang++ returns str like 0.0.0git, others return 0.0.0
    numeric_version = [re.sub(r"\D", "", v) for v in version]

    if tuple(map(int, numeric_version)) >= minimum_required_version:
        return (True, TorchVersion(".".join(numeric_version)))

    compiler = f"{compiler} {'.'.join(numeric_version)}"
    logger.warning(ABI_INCOMPATIBILITY_WARNING, compiler)

    return (False, TorchVersion(".".join(numeric_version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if IS_WINDOWS else "nvcc")
    if not os.path.exists(nvcc):
        raise FileNotFoundError(
            f"nvcc not found at '{nvcc}'. Ensure CUDA path '{CUDA_HOME}' is correct."
        )

    cuda_version_str = (
        subprocess.check_output([nvcc, "--version"])
        .strip()
        .decode(*SUBPROCESS_DECODE_ARGS)
    )
    cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)
    if cuda_version is None:
        return

    cuda_str_version = cuda_version.group(1)
    cuda_ver = Version(cuda_str_version)
    if torch.version.cuda is None:
        return

    torch_cuda_version = Version(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(
                CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda
            )
        logger.warning(CUDA_MISMATCH_WARN, cuda_str_version, torch.version.cuda)

    if not (
        sys.platform.startswith("linux")
        and os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI")
        not in ["ON", "1", "YES", "TRUE", "Y"]
        and _is_binary_build()
    ):
        return

    cuda_compiler_bounds: VersionMap = (
        CUDA_CLANG_VERSIONS if compiler_name.startswith("clang") else CUDA_GCC_VERSIONS
    )

    if cuda_str_version not in cuda_compiler_bounds:
        logger.warning(
            "There are no %s version bounds defined for CUDA version %s",
            compiler_name,
            cuda_str_version,
        )
    else:
        min_compiler_version, max_excl_compiler_version = cuda_compiler_bounds[
            cuda_str_version
        ]
        # Special case for 11.4.0, which has lower compiler bounds than 11.4.1
        if "V11.4.48" in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        min_compiler_version_str = ".".join(map(str, min_compiler_version))
        max_excl_compiler_version_str = ".".join(map(str, max_excl_compiler_version))

        version_bound_str = (
            f">={min_compiler_version_str}, <{max_excl_compiler_version_str}"
        )

        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(
                f"The current installed version of {compiler_name} ({compiler_version}) is less "
                f"than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). "
                f"Please make sure to use an adequate version of {compiler_name} ({version_bound_str})."
            )
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(
                f"The current installed version of {compiler_name} ({compiler_version}) is greater "
                f"than the maximum required version by CUDA {cuda_str_version}. "
                f"Please make sure to use an adequate version of {compiler_name} ({version_bound_str})."
            )


# Specify Visual Studio C runtime library for hipcc
def _set_hipcc_runtime_lib(is_standalone, debug) -> None:
    if is_standalone:
        if debug:
            COMMON_HIP_FLAGS.append("-fms-runtime-lib=static_dbg")
        else:
            COMMON_HIP_FLAGS.append("-fms-runtime-lib=static")
    else:
        if debug:
            COMMON_HIP_FLAGS.append("-fms-runtime-lib=dll_dbg")
        else:
            COMMON_HIP_FLAGS.append("-fms-runtime-lib=dll")


def _append_sycl_std_if_no_std_present(cflags) -> None:
    if not any(flag.startswith("-sycl-std=") for flag in cflags):
        cflags.append("-sycl-std=2020")


def _wrap_sycl_host_flags(cflags):
    host_cflags = []
    host_cxx = get_cxx_compiler()
    if IS_WINDOWS:
        for flag in cflags:
            if flag.startswith("-I"):
                flag = flag.replace("\\", "\\\\").replace("-I", "/I")
            else:
                flag = flag.replace("-D", "/D")
            flag = flag.replace('"', '\\"')
            host_cflags.append(flag)
        joined_host_cflags = " ".join(host_cflags)

        external_include = _join_sycl_home("include").replace("\\", "\\\\")

        # Some versions of DPC++ compiler pass paths to SYCL headers as user include paths (`-I`) rather
        # than system paths (`-isystem`). This makes host compiler to report warnings encountered in the
        # SYCL headers, such as deprecated warnings, even if warmed API is not actually used in the program.
        # We expect that this issue will be addressed in the later version of DPC++ compiler. To workaround the
        # issue now we wrap paths to SYCL headers in `/external:I`. Warning free compilation is especially important
        # for Windows build as `/sdl` compilation flag assumes that and we will fail compilation otherwise.
        wrapped_host_cflags = [
            f"-fsycl-host-compiler={host_cxx}",
            f'-fsycl-host-compiler-options="\\"/external:I{external_include}\\" /external:W0 {joined_host_cflags}"',
        ]
    else:
        joined_host_cflags = " ".join(cflags)
        wrapped_host_cflags = [
            f"-fsycl-host-compiler={host_cxx}",
            shlex.quote(f"-fsycl-host-compiler-options={joined_host_cflags}"),
        ]
    return wrapped_host_cflags


def _join_cuda_home(*paths) -> str:
    """
    Join paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    """
    if CUDA_HOME is None:
        raise OSError(
            "CUDA_HOME environment variable is not set. "
            "Please set it to your CUDA install root."
        )
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path: str) -> bool:
    valid_ext = [".cu", ".cuh"]
    if IS_HIP_EXTENSION:
        valid_ext.append(".hip")
    return os.path.splitext(path)[1] in valid_ext


def _is_sycl_file(path: str) -> bool:
    valid_ext = [".sycl"]
    return os.path.splitext(path)[1] in valid_ext


def include_paths(device_type: str = "cpu", torch_include_dirs=True) -> list[str]:
    """
    Get the include paths required to build a C++ or CUDA or SYCL extension.

    Args:
        device_type: Defaults to "cpu".
    Returns:
        A list of include path strings.
    """
    paths = []
    lib_include = os.path.join(_TORCH_PATH, "include")
    if torch_include_dirs:
        paths.extend(
            [
                lib_include,
                # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
                os.path.join(lib_include, "torch", "csrc", "api", "include"),
            ]
        )
    if device_type == "cuda" and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, "THH"))
        paths.append(_join_rocm_home("include"))
    elif device_type == "cuda":
        cuda_home_include = _join_cuda_home("include")
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != "/usr/include":
            paths.append(cuda_home_include)

        # Support CUDA_INC_PATH env variable supported by CMake files
        if (
            cuda_inc_path := os.environ.get("CUDA_INC_PATH", None)
        ) and cuda_inc_path != "/usr/include":
            paths.append(cuda_inc_path)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, "include"))
    elif device_type == "xpu":
        paths.append(_join_sycl_home("include"))
        paths.append(_join_sycl_home("include", "sycl"))
    return paths


def library_paths(
    device_type: str = "cpu",
    torch_include_dirs: bool = True,
    cross_target_platform: str | None = None,
) -> list[str]:
    """
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        device_type: Defaults to "cpu".

    Returns:
        A list of library path strings.
    """

    paths = []

    if torch_include_dirs:
        # We need to link against libtorch.so
        paths.extend([TORCH_LIB_PATH])

    if device_type == "cuda" and IS_HIP_EXTENSION:
        lib_dir = "lib"
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, "lib"))
    elif device_type == "cuda":
        if cross_target_platform == "windows":
            lib_dir = os.path.join("lib", "x64")
            if WINDOWS_CUDA_HOME is None:
                raise RuntimeError(
                    "Need to set WINDOWS_CUDA_HOME for windows cross-compilation"
                )
            paths.append(os.path.join(WINDOWS_CUDA_HOME, lib_dir))
        else:
            if IS_WINDOWS:
                lib_dir = os.path.join("lib", "x64")
            else:
                lib_dir = "lib64"
                if not os.path.exists(_join_cuda_home(lib_dir)) and os.path.exists(
                    _join_cuda_home("lib")
                ):
                    # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                    # Note that it's also possible both don't exist (see
                    # _find_cuda_home) - in that case we stay with 'lib64'.
                    lib_dir = "lib"

            paths.append(_join_cuda_home(lib_dir))
            if CUDNN_HOME is not None:
                paths.append(os.path.join(CUDNN_HOME, lib_dir))
    elif device_type == "xpu":
        if IS_WINDOWS:
            lib_dir = os.path.join("lib", "x64")
        else:
            lib_dir = "lib64"
            if not os.path.exists(_join_sycl_home(lib_dir)) and os.path.exists(
                _join_sycl_home("lib")
            ):
                lib_dir = "lib"

        paths.append(_join_sycl_home(lib_dir))

    return paths


def is_ninja_available() -> bool:
    """Return ``True`` if the `ninja <https://ninja-build.org/>`_ build system is available on the system, ``False`` otherwise."""
    try:
        subprocess.check_output(["ninja", "--version"])
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability() -> None:
    """Raise ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not available on the system, does nothing otherwise."""
    if not is_ninja_available():
        raise RuntimeError(
            "Ninja is required to load C++ extensions (pip install ninja to get it)"
        )


@deprecated(
    "PyBind11 ABI handling is internal to PyBind11; this will be removed after PyTorch 2.9.0"
)
def _get_pybind11_abi_build_flags() -> list[str]:
    return []


def check_compiler_is_gcc(compiler) -> bool:
    if not IS_LINUX:
        return False

    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except (subprocess.CalledProcessError, OSError):
        try:
            version_string = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT, env=env
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except (subprocess.CalledProcessError, OSError):
            return False
    # Check for GCC by verifying both COLLECT_GCC and gcc version string are present
    # This works for c++, g++, gcc, and versioned variants like g++-13
    pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
    has_collect_gcc = pattern.search(version_string) is not None
    if has_collect_gcc and "gcc version" in version_string:
        return True
    return False
