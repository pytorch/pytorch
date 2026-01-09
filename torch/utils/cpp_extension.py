# mypy: allow-untyped-defs
import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import types
import collections
from pathlib import Path
import errno
import logging

logger = logging.getLogger(__name__)

import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from typing_extensions import deprecated
from torch.torch_version import TorchVersion, Version


from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
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
    '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
    '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.4': ((6, 0, 0), (12, 0)),
    '11.5': ((6, 0, 0), (12, 0)),
    '11.6': ((6, 0, 0), (12, 0)),
    '11.7': ((6, 0, 0), (12, 0)),
    '12.0': ((6, 0, 0), (13, 0)),
    '12.1': ((6, 0, 0), (13, 0)),
    '12.2': ((6, 0, 0), (13, 0)),
    '12.3': ((6, 0, 0), (14, 0)),
    '12.4': ((6, 0, 0), (14, 0)),
    '12.5': ((6, 0, 0), (14, 0)),
    '12.6': ((6, 0, 0), (14, 0)),
    '12.7': ((6, 0, 0), (14, 0)),
    '12.8': ((6, 0, 0), (14, 0)),
    '12.9': ((6, 0, 0), (14, 0)),
    '13.0': ((6, 0, 0), (16, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {
    '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
    '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
    '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
    '12.0': ((7, 0), (15, 0)),
    '12.1': ((7, 0), (15, 0)),
    '12.2': ((7, 0), (16, 0)),
    '12.3': ((7, 0), (16, 0)),
    '12.4': ((7, 0), (17, 0)),
    '12.5': ((7, 0), (18, 0)),
    '12.6': ((7, 0), (18, 0)),
    '12.7': ((7, 0), (19, 0)),
    '12.8': ((7, 0), (19, 0)),
    '12.9': ((7, 0), (19, 0)),
    '13.0': ((7, 0), (21, 0)),
}

__all__ = ["get_default_build_root", "check_compiler_ok_for_platform", "get_compiler_abi_compatibility_and_version", "BuildExtension",
           "CppExtension", "CUDAExtension", "SyclExtension", "include_paths", "library_paths", "load", "load_inline", "is_ninja_available",
           "verify_ninja_availability", "remove_extension_h_precompiler_headers", "get_cxx_compiler", "check_compiler_is_gcc"]
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
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]

def _find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        logger.warning("No CUDA runtime is found, using CUDA_HOME='%s'", cuda_home)
    return cuda_home

def _find_rocm_home() -> str | None:
    """Find the ROCm install path."""
    # Guess #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Guess #2
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(
                os.path.realpath(hipcc_path)))
            # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            # Guess #3
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        logger.warning("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
    return rocm_home

def _find_sycl_home() -> str | None:
    sycl_home = None
    icpx_path = shutil.which('icpx')
    # Guess 1: for source code build developer/user, we'll have icpx in PATH,
    # which will tell us the SYCL_HOME location.
    if icpx_path is not None:
        sycl_home = os.path.dirname(os.path.dirname(
            os.path.realpath(icpx_path)))

    # Guess 2: for users install Pytorch with XPU support, the sycl runtime is
    # inside intel-sycl-rt, which is automatically installed via pip dependency.
    else:
        try:
            files = importlib.metadata.files('intel-sycl-rt') or []
            for f in files:
                if f.name == "libsycl.so":
                    sycl_home = os.path.dirname(Path(f.locate()).parent.resolve())
                    break
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Trying to find SYCL_HOME from intel-sycl-rt package, but it is not installed.")
    return sycl_home

def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. '
                      'Please set it to your ROCm install root.')
    return os.path.join(ROCM_HOME, *paths)

def _join_sycl_home(*paths) -> str:
    """
    Join paths with SYCL_HOME, or raises an error if it SYCL_HOME is not found.

    This is basically a lazy way of raising an error for missing SYCL_HOME
    only once we need to get any SYCL-specific path.
    """
    if SYCL_HOME is None:
        raise OSError('SYCL runtime is not dected. Please setup the pytorch '
                      'prerequisites for Intel GPU following the instruction in '
                      'https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support '
                      'or install intel-sycl-rt via pip.')

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
CUDA_MISMATCH_WARN = (
    "The detected CUDA version (%s) has a minor version mismatch with the version that was used to compile PyTorch (%s). Most likely this shouldn't be a problem."
)
CUDA_NOT_FOUND_MESSAGE = (
    "CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH"
    "environment variable or add NVCC to your system PATH. The extension compilation will fail."
)
ROCM_HOME = _find_rocm_home() if (torch.cuda._is_compiled() and torch.version.hip) else None
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = bool(ROCM_HOME is not None and torch.version.hip is not None)
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

CUDA_HOME = _find_cuda_home() if (torch.cuda._is_compiled() and torch.version.cuda) else None
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
SYCL_HOME = _find_sycl_home() if torch.xpu._is_compiled() else None
WINDOWS_CUDA_HOME = os.environ.get('WINDOWS_CUDA_HOME')  # used for AOTI cross-compilation

# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']

MSVC_IGNORE_CUDAFE_WARNINGS = [
    'base_class_has_different_dll_interface',
    'field_without_dll_interface',
    'dll_interface_conflict_none_assumed',
    'dll_interface_conflict_dllexport_assumed'
]

COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

COMMON_HIP_FLAGS = [
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
    '-DHIPBLAS_V2',
]

if not IS_WINDOWS:
    COMMON_HIP_FLAGS.append('-fPIC')

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
    '-DHIP_ENABLE_WARP_SYNC_BUILTINS=1'
]

if IS_WINDOWS:
    # Compatibility flags, similar to those set in cmake/Dependencies.cmake.
    COMMON_HIPCC_FLAGS.append('-fms-extensions')
    # Suppress warnings about dllexport.
    COMMON_HIPCC_FLAGS.append('-Wno-ignored-attributes')


def _get_icpx_version() -> str:
    icpx = 'icx' if IS_WINDOWS else 'icpx'
    compiler_info = subprocess.check_output([icpx, '--version'])
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())
    version = ['0', '0', '0'] if match is None else list(match.groups())
    version = list(map(int, version))
    if len(version) != 3:
        raise AssertionError("Failed to parse DPC++ compiler version")
    # Aligning version format with what torch.version.xpu() returns
    return f"{version[0]}{version[1]:02}{version[2]:02}"


def _get_sycl_arch_list():
    if 'TORCH_XPU_ARCH_LIST' in os.environ:
        return os.environ.get('TORCH_XPU_ARCH_LIST')
    arch_list = torch.xpu.get_arch_list()
    # Dropping dg2* archs since they lack hardware support for fp64 and require
    # special consideration from the user. If needed these platforms can
    # be requested thru TORCH_XPU_ARCH_LIST environment variable.
    arch_list = [x for x in arch_list if not x.startswith('dg2')]
    return ','.join(arch_list)


# If arch list returned by _get_sycl_arch_list() is empty, then sycl kernels will be compiled
# for default spir64 target and avoid device specific compilations entirely. Further, kernels
# will be JIT compiled at runtime.
def _append_sycl_targets_if_missing(cflags) -> None:
    if any(flag.startswith('-fsycl-targets=') for flag in cflags):
        # do nothing: user has manually specified sycl targets
        return
    if _get_sycl_arch_list() != '':
        # AOT (spir64_gen) + JIT (spir64)
        cflags.append('-fsycl-targets=spir64_gen,spir64')
    else:
        # JIT (spir64)
        cflags.append('-fsycl-targets=spir64')

def _get_sycl_device_flags(cflags):
    # We need last occurrence of -fsycl-targets as it will be the one taking effect.
    # So searching in reversed list.
    flags = [f for f in reversed(cflags) if f.startswith('-fsycl-targets=')]
    if not flags:
        raise AssertionError("bug: -fsycl-targets should have been amended to cflags")

    arch_list = _get_sycl_arch_list()
    if arch_list != '':
        flags += [f'-Xs "-device {arch_list}"']
    return flags

_COMMON_SYCL_FLAGS = [
    '-fsycl',
]

_SYCL_DLINK_FLAGS = [
    *_COMMON_SYCL_FLAGS,
    '-fsycl-link',
    '--offload-compress',
]

JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}

min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode

def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> list[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']

def _maybe_write(filename, new_content) -> None:
    r'''
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    '''
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, 'w') as source_file:
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
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


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
    env['LC_ALL'] = 'C'  # Don't localize output
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except subprocess.CalledProcessError:
        # If '-v' fails, try '--version'
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            # Though on Ubuntu it's sometimes called "Ubuntu clang version"
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
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
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        logger.warning(WRONG_COMPILER_WARNING, compiler, _accepted_compilers_for_platform()[0], sys.platform, _accepted_compilers_for_platform()[0], compiler, compiler)
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            compiler_info = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
        version = ['0', '0', '0'] if match is None else list(match.groups())
    except (subprocess.CalledProcessError, OSError):
        logger.warning('Error checking compiler version for %s', compiler, exc_info=True)
        return (False, TorchVersion('0.0.0'))

    # convert alphanumeric string to numeric string
    # amdclang++ returns str like 0.0.0git, others return 0.0.0
    numeric_version = [re.sub(r'\D', '', v) for v in version]

    if tuple(map(int, numeric_version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(numeric_version)))

    compiler = f'{compiler} {".".join(numeric_version)}'
    logger.warning(ABI_INCOMPATIBILITY_WARNING, compiler)

    return (False, TorchVersion('.'.join(numeric_version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
    if not os.path.exists(nvcc):
        raise FileNotFoundError(f"nvcc not found at '{nvcc}'. Ensure CUDA path '{CUDA_HOME}' is correct.")

    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
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
            raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
        logger.warning(CUDA_MISMATCH_WARN, cuda_str_version, torch.version.cuda)

    if not (sys.platform.startswith('linux') and
            os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and
            _is_binary_build()):
        return

    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS

    if cuda_str_version not in cuda_compiler_bounds:
        logger.warning('There are no %s version bounds defined for CUDA version %s', compiler_name, cuda_str_version)
    else:
        min_compiler_version, max_excl_compiler_version = cuda_compiler_bounds[cuda_str_version]
        # Special case for 11.4.0, which has lower compiler bounds than 11.4.1
        if "V11.4.48" in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        min_compiler_version_str = '.'.join(map(str, min_compiler_version))
        max_excl_compiler_version_str = '.'.join(map(str, max_excl_compiler_version))

        version_bound_str = f'>={min_compiler_version_str}, <{max_excl_compiler_version_str}'

        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is less '
                f'than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is greater '
                f'than the maximum required version by CUDA {cuda_str_version}. '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )


# Specify Visual Studio C runtime library for hipcc
def _set_hipcc_runtime_lib(is_standalone, debug) -> None:
    if is_standalone:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static')
    else:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll')

def _append_sycl_std_if_no_std_present(cflags) -> None:
    if not any(flag.startswith('-sycl-std=') for flag in cflags):
        cflags.append('-sycl-std=2020')


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
        joined_host_cflags = ' '.join(host_cflags)

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
        joined_host_cflags = ' '.join(cflags)
        wrapped_host_cflags = [
            f"-fsycl-host-compiler={host_cxx}",
            shlex.quote(f"-fsycl-host-compiler-options={joined_host_cflags}"),
        ]
    return wrapped_host_cflags


class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/CUDA/SYCL compilation (and support for CUDA/SYCL files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages/compilers (the only expected values are ``cxx``, ``nvcc`` or
    ``sycl``) to a list of additional compiler flags to supply to the compiler.
    This makes it possible to supply different flags to the C++, CUDA and SYCL
    compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options."""
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '%s. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                logger.warning(msg, 'we could not find ninja.')
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        sycl_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not (cuda_ext and sycl_ext) and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                elif ext == '.sycl':
                    sycl_ext = True

                # This check accounts on a case when cuda and sycl sources
                # are mixed in the same extension. We can stop checking
                # sources if both are found or there is no more sources.
                if cuda_ext and sycl_ext:
                    break

            extension = next(extension_iter, None)

        if sycl_ext:
            if not self.use_ninja:
                raise AssertionError("ninja is required to build sycl extensions.")

        if cuda_ext and not IS_HIP_EXTENSION:
            _check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx', 'nvcc' and 'sycl' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx', 'nvcc' or 'sycl' is
            # passed to extra_compile_args in CUDAExtension or SyclExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc', 'sycl']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')

            if IS_HIP_EXTENSION:
                self._hipify_compile_flags(extension)

            if extension.py_limited_api:
                # compile any extension that has passed in py_limited_api to the
                # Extension constructor with the Py_LIMITED_API flag set to our
                # min supported CPython version.
                # See https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
                self._add_compile_flag(extension, f'-DPy_LIMITED_API={min_supported_cpython}')
            self._define_torch_extension_name(extension)

            if 'nvcc_dlink' in extension.extra_compile_args:
                if not self.use_ninja:
                    raise AssertionError(
                        f"With dlink=True, ninja is required to build cuda extension {extension.name}."
                    )

        # Register .cu, .cuh, .hip, .mm and .sycl as valid source extensions.
        # NOTE: At the moment .sycl is not a standard extension for SYCL supported
        # by compiler. Here we introduce a torch level convention that SYCL sources
        # should have .sycl file extension.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip', '.sycl']
        if torch.backends.mps.is_built():
            self.compiler.src_extensions += ['.mm']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags + _get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                _ccbin is not None
                and not any(flag.startswith(('-ccbin', '--compiler-bindir')) for flag in cflags)
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths) -> None:
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if IS_HIP_EXTENSION:
                        cflags = COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                if IS_HIP_EXTENSION:
                    cflags = COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567  # codespell:ignore
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc/sycl to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
                cuda_dlink_post_cflags = [shlex.quote(f) for f in cuda_dlink_post_cflags]
            else:
                cuda_dlink_post_cflags = None

            sycl_post_cflags = None
            sycl_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = extra_cc_cflags + common_cflags + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs['sycl']
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = extra_cc_cflags + common_cflags + post_cflags
                append_std17_if_no_std_present(host_cflags)
                # escaping quoted arguments to pass them thru SYCL compiler
                icpx_version = _get_icpx_version()
                if int(icpx_version) >= 20250200:
                    host_cflags = [item.replace('"', '\\"') for item in host_cflags]
                else:
                    host_cflags = [item.replace('"', '\\\\"') for item in host_cflags]
                # Note the order: shlex.quote sycl_flags first, _wrap_sycl_host_flags
                # second. Reason is that sycl host flags are quoted, space containing
                # strings passed to SYCL compiler.
                sycl_cflags = [shlex.quote(f) for f in sycl_cflags]
                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = [shlex.quote(f) for f in sycl_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (COMMON_NVCC_FLAGS +
                    cflags + _get_cuda_arch_flags(cflags))

        def win_hip_flags(cflags):
            return (COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + _get_rocm_arch_flags(cflags))

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')  # codespell:ignore
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        if IS_HIP_EXTENSION:
                            nvcc = _get_hipcc_path()
                        else:
                            nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        if IS_HIP_EXTENSION:
                            cflags = win_hip_flags(cflags)
                        else:
                            cflags = win_cuda_flags(cflags) + ['-std=c++17', '--use-local-env']
                            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                                cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources,
                                   output_dir=None,
                                   macros=None,
                                   include_dirs=None,
                                   debug=0,
                                   extra_preargs=None,
                                   extra_postargs=None,
                                   depends=None,
                                   is_standalone=False):
            if not self.compiler.initialized:
                self.compiler.initialize()
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any.
            # For ninja build, the build location is not local, but instead, the build happens
            # in a script-created build folder. Thus, relative paths lose their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here.
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            # Replace space with \ when using hipcc (hipcc passes includes to clang without ""s so clang sees space in include paths as new argument)
            if IS_HIP_EXTENSION:
                pp_opts = ["-I{}".format(s[2:].replace(" ", "\\")) if s.startswith('-I') else s for s in pp_opts]
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            cflags = cflags + common_cflags + pp_opts + COMMON_MSVC_FLAGS
            if IS_HIP_EXTENSION:
                _set_hipcc_runtime_lib(is_standalone, debug)
                common_cflags.extend(COMMON_HIP_FLAGS)
            else:
                common_cflags.extend(COMMON_MSVC_FLAGS)
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['-std=c++17']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                if not IS_HIP_EXTENSION:
                    cuda_cflags.append('--use-local-env')
                    for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                        cuda_cflags.append('-Xcudafe')
                        cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = win_hip_flags(cuda_post_cflags)
                else:
                    cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None

            sycl_cflags = None
            sycl_post_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = common_cflags + pp_opts + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs['sycl']
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = common_cflags + pp_opts + post_cflags
                append_std17_if_no_std_present(host_cflags)

                sycl_cflags = _nt_quote_args(sycl_cflags)
                host_cflags = _nt_quote_args(host_cflags)

                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = _nt_quote_args(sycl_post_cflags)


            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl)

            # Return *all* object filenames, not just the ones we just built.
            return objects
        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511  # codespell:ignore
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Remove ABI component only if it actually exists in a file name, see gh-170542.
            if len(ext_filename_parts) > 2:
                # Omit the second to last element.
                without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                ext_filename = '.'.join(without_abi)
        return ext_filename

    def get_export_symbols(self, ext):
        if IS_WINDOWS:
            # Skips exporting the module "PyInit_" function that the
            # distutils Extension.get_export_symbols would add to
            # ext.export_symbols. Only relevant for Windows builds.
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def _check_abi(self) -> tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                   'This may lead to multiple activations of the VC env.'
                   'Please set `DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag) -> None:
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    # Simple hipify, replace the first occurrence of CUDA with HIP
    # in flags starting with "-" and containing "CUDA", but exclude -I flags
    def _hipify_compile_flags(self, extension) -> None:
        if isinstance(extension.extra_compile_args, dict) and 'nvcc' in extension.extra_compile_args:
            modified_flags = []
            for flag in extension.extra_compile_args['nvcc']:
                if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                    # check/split flag into flag and value
                    parts = flag.split("=", 1)
                    if len(parts) == 2:
                        flag_part, value_part = parts
                        # replace fist instance of "CUDA" with "HIP" only in the flag and not flag value
                        modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                        modified_flag = f"{modified_flag_part}={value_part}"
                    else:
                        # replace fist instance of "CUDA" with "HIP" in flag
                        modified_flag = flag.replace("CUDA", "HIP", 1)
                    modified_flags.append(modified_flag)
                    logger.info('Modified flag: %s -> %s', flag, modified_flag)
                else:
                    modified_flags.append(flag)
            extension.extra_compile_args['nvcc'] = modified_flags

    def _define_torch_extension_name(self, extension) -> None:
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)


def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
        ...     name='extension',
        ...     ext_modules=[
        ...         CppExtension(
        ...             name='extension',
        ...             sources=['extension.cpp'],
        ...             extra_compile_args=['-g'],
        ...             extra_link_args=['-Wl,--no-as-needed', '-lm'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_WINDOWS:
        libraries.append("sleef")

    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
        ...     name='cuda_extension',
        ...     ext_modules=[
        ...         CUDAExtension(
        ...                 name='cuda_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cu'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'nvcc': ['-O2']},
        ...                 extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, PyTorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``
    ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``

    The +PTX option causes extension kernel binaries to include PTX instructions for the specified
    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=
    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with
    CC >= 8.6). This improves your binary's forward compatibility. However, relying on older PTX to
    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on
    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you're always better
    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,
    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but
    "8.0 8.6" would be better.

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows.
    To workaround the issue, move python binding logic to pure C++ file.

    Example use:
        #include <ATen/ATen.h>
        at::Tensor SigmoidAlphaBlendForwardCuda(....)

    Instead of:
        #include <torch/extension.h>
        torch::Tensor SigmoidAlphaBlendForwardCuda(...)

    Currently open issue for nvcc bug: https://github.com/pytorch/pytorch/issues/69460
    Complete workaround code example: https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48

    Relocatable device code linking:

    If you want to reference device symbols across compilation units (across object files),
    the object files need to be built with `relocatable device code` (-rdc=true or -dc).
    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.
    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.
    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step
    helps reduce the protentional perf degradation of `-rdc`.
    Note that it needs to be used at both steps to be useful.

    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.
    There is also a case where `-dlink` is used without `-rdc`:
    when an extension is linked against a static lib containing rdc-compiled objects
    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).

    Note: Ninja is required to build a CUDA Extension with RDC linking.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> CUDAExtension(
        ...        name='cuda_extension',
        ...        sources=['extension.cpp', 'extension_kernel.cu'],
        ...        dlink=True,
        ...        dlink_libraries=["dlink_lib"],
        ...        extra_compile_args={'cxx': ['-g'],
        ...                            'nvcc': ['-O2', '-rdc=true']})
    """
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(device_type="cuda")
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        libraries.append('amdhip64')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    if IS_HIP_EXTENSION:
        from .hipify import hipify_python
        build_dir = os.getcwd()
        hipify_result = hipify_python.hipify(
            project_directory=build_dir,
            output_directory=build_dir,
            header_include_dirs=include_dirs,
            includes=[os.path.join(build_dir, '*')],  # limit scope to build_dir only
            extra_files=[os.path.abspath(s) for s in sources],
            show_detailed=True,
            is_pytorch_extension=True,
            hipify_extra_files_only=True,  # don't hipify everything in includes path
        )

        hipified_sources = set()
        for source in sources:
            s_abs = os.path.abspath(source)
            hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                              hipify_result[s_abs].hipified_path is not None) else s_abs)
            # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
            # *never* absolute paths
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

        sources = list(hipified_sources)

    include_dirs += include_paths(device_type="cuda")
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']   # Device Link Time Optimization started from cuda 11.2

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


def SyclExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for SYCL/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a SYCL/C++
    extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import BuildExtension, SyclExtension
        >>> setup(
        ...     name='xpu_extension',
        ...     ext_modules=[
        ...     SyclExtension(
        ...                 name='xpu_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cpp'],
        ...                 extra_compile_args={'cxx': ['-g', '-std=c++20', '-fPIC']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension. If down the road a new card is installed the
    extension may need to be recompiled. You can override the default behavior using
    `TORCH_XPU_ARCH_LIST` to explicitly specify which device architectures you want the extension
    to support:

    ``TORCH_XPU_ARCH_LIST="pvc,xe-lpg" python build_my_extension.py``

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note: Ninja is required to build SyclExtension.
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("c10_xpu")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("sycl")
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    libraries.append("torch_xpu")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths(device_type="xpu")
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)

def include_paths(device_type: str = "cpu", torch_include_dirs=True) -> list[str]:
    """
    Get the include paths required to build a C++ or CUDA or SYCL extension.

    Args:
        device_type: Defaults to "cpu".
    Returns:
        A list of include path strings.
    """
    paths = []
    lib_include = os.path.join(_TORCH_PATH, 'include')
    if torch_include_dirs:
        paths.extend([
            lib_include,
            # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
            os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        ])
    if device_type == "cuda" and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    elif device_type == "cuda":
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)

        # Support CUDA_INC_PATH env variable supported by CMake files
        if (cuda_inc_path := os.environ.get("CUDA_INC_PATH", None)) and \
                cuda_inc_path != '/usr/include':

            paths.append(cuda_inc_path)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    elif device_type == "xpu":
        paths.append(_join_sycl_home('include'))
        paths.append(_join_sycl_home('include', 'sycl'))
    return paths


def library_paths(device_type: str = "cpu", torch_include_dirs: bool = True, cross_target_platform: str | None = None) -> list[str]:
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
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    elif device_type == "cuda":
        if cross_target_platform == "windows":
            lib_dir = os.path.join('lib', 'x64')
            if WINDOWS_CUDA_HOME is None:
                raise RuntimeError("Need to set WINDOWS_CUDA_HOME for windows cross-compilation")
            paths.append(os.path.join(WINDOWS_CUDA_HOME, lib_dir))
        else:
            if IS_WINDOWS:
                lib_dir = os.path.join('lib', 'x64')
            else:
                lib_dir = 'lib64'
                if (not os.path.exists(_join_cuda_home(lib_dir)) and
                        os.path.exists(_join_cuda_home('lib'))):
                    # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                    # Note that it's also possible both don't exist (see
                    # _find_cuda_home) - in that case we stay with 'lib64'.
                    lib_dir = 'lib'

            paths.append(_join_cuda_home(lib_dir))
            if CUDNN_HOME is not None:
                paths.append(os.path.join(CUDNN_HOME, lib_dir))
    elif device_type == "xpu":
        if IS_WINDOWS:
            lib_dir = os.path.join('lib', 'x64')
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_sycl_home(lib_dir)) and
                    os.path.exists(_join_sycl_home('lib'))):
                lib_dir = 'lib'

        paths.append(_join_sycl_home(lib_dir))

    return paths


def load(name,
         sources: str | list[str],
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_sycl_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False,
         with_cuda: bool | None = None,
         with_sycl: bool | None = None,
         is_python_module=True,
         is_standalone=False,
         keep_intermediates=True):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.

    SYCL support with mixed compilation is provided. Simply pass SYCL source
    files (``.sycl``) along with other sources. Such files will be detected
    and compiled with SYCL compiler (such as Intel DPC++ Compiler) rather
    than the C++ compiler. You can pass additional flags to SYCL compiler
    via ``extra_sycl_cflags``, just like with ``extra_cflags`` for C++.
    SYCL compiler is expected to be found via system PATH environment
    variable.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_sycl_cflags: optional list of compiler flags to forward to SYCL
            compiler when building SYCL sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.cu`` or
            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.sycl`` in
            ``sources``. Set it to `True`` to force SYCL headers and
            libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, behavior depends on
            ``is_standalone``.
        is_standalone: If ``False`` (default) loads the constructed extension
            into the process as a plain dynamic library. If ``True``, build a
            standalone executable.

    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Return the path to the executable. (On Windows, TORCH_LIB_PATH is
            added to the PATH environment variable as a side effect.)

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
        ...     name='extension',
        ...     sources=['extension.cpp', 'extension_kernel.cu'],
        ...     extra_cflags=['-O2'],
        ...     verbose=True)
    """
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory or _get_build_directory(name, verbose),
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates)

@deprecated("PyBind11 ABI handling is internal to PyBind11; this will be removed after PyTorch 2.9.0")
def _get_pybind11_abi_build_flags() -> list[str]:
    return []

def check_compiler_is_gcc(compiler) -> bool:
    if not IS_LINUX:
        return False

    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except (subprocess.CalledProcessError, OSError):
        try:
            version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
        except (subprocess.CalledProcessError, OSError):
            return False
    # Check for GCC by verifying both COLLECT_GCC and gcc version string are present
    # This works for c++, g++, gcc, and versioned variants like g++-13
    pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
    has_collect_gcc = pattern.search(version_string) is not None
    if has_collect_gcc and 'gcc version' in version_string:
        return True
    return False

def _check_and_build_extension_h_precompiler_headers(
        extra_cflags,
        extra_include_paths,
        is_standalone=False) -> None:
    r'''
    Precompiled Headers(PCH) can pre-build the same headers and reduce build time for pytorch load_inline modules.
    GCC official manual: https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html
    PCH only works when built pch file(header.h.gch) and build target have the same build parameters. So, We need
    add a signature file to record PCH file parameters. If the build parameters(signature) changed, it should rebuild
    PCH file.

    Note:
    1. Windows and MacOS have different PCH mechanism. We only support Linux currently.
    2. It only works on GCC/G++.
    '''
    if not IS_LINUX:
        return

    compiler = get_cxx_compiler()

    b_is_gcc = check_compiler_is_gcc(compiler)
    if b_is_gcc is False:
        return

    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    def listToString(s):
        # initialize an empty string
        string = ""
        if s is None:
            return string

        # traverse in the string
        for element in s:
            string += (element + ' ')
        # return string
        return string

    def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
        return re.sub(
            r"[ \n]+",
            " ",
            f"""
                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}
            """,
        ).strip()

    def command_to_signature(cmd):
        signature = cmd.replace(' ', '_')
        return signature

    def check_pch_signature_in_file(file_path, signature):
        b_exist = os.path.isfile(file_path)
        if b_exist is False:
            return False

        with open(file_path) as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            return signature == content

    def _create_if_not_exist(path_dir) -> None:
        if not os.path.exists(path_dir):
            try:
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f"Fail to create path {path_dir}") from exc

    def write_pch_signature_to_file(file_path, pch_sign) -> None:
        _create_if_not_exist(os.path.dirname(file_path))
        with open(file_path, "w") as f:
            f.write(pch_sign)
            f.close()

    def build_precompile_header(pch_cmd) -> None:
        try:
            subprocess.check_output(shlex.split(pch_cmd), stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compile PreCompile Header fail, command: {pch_cmd}") from e

    extra_cflags_str = listToString(extra_cflags)
    extra_include_paths_str = " ".join(
        [f"-I{include}" for include in extra_include_paths] if extra_include_paths else []
    )

    lib_include = os.path.join(_TORCH_PATH, 'include')
    torch_include_dirs = [
        f"-I {lib_include}",
        # Python.h
        "-I {}".format(sysconfig.get_path("include")),
        # torch/all.h
        "-I {}".format(os.path.join(lib_include, 'torch', 'csrc', 'api', 'include')),
    ]

    torch_include_dirs_str = listToString(torch_include_dirs)

    common_cflags = []
    if not is_standalone:
        common_cflags += ['-DTORCH_API_INCLUDE_EXTENSION_H']

    common_cflags += ['-std=c++17', '-fPIC']
    common_cflags_str = listToString(common_cflags)

    pch_cmd = format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags_str, torch_include_dirs_str, extra_cflags_str, extra_include_paths_str)
    pch_sign = command_to_signature(pch_cmd)

    if os.path.isfile(head_file_pch) is not True:
        build_precompile_header(pch_cmd)
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        if b_same_sign is False:
            build_precompile_header(pch_cmd)
            write_pch_signature_to_file(head_file_signature, pch_sign)

def remove_extension_h_precompiler_headers() -> None:
    def _remove_if_file_exists(path_file) -> None:
        if os.path.exists(path_file):
            os.remove(path_file)

    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    _remove_if_file_exists(head_file_pch)
    _remove_if_file_exists(head_file_signature)

def load_inline(name,
                cpp_sources,
                cuda_sources=None,
                sycl_sources=None,
                functions=None,
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_sycl_cflags=None,
                extra_ldflags=None,
                extra_include_paths=None,
                build_directory=None,
                verbose=False,
                with_cuda=None,
                with_sycl=None,
                is_python_module=True,
                with_pytorch_error_handling=True,
                keep_intermediates=True,
                use_pch=False,
                no_implicit_headers=False):
    r'''
    Load a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``
    file and  prepended with ``torch/types.h``, ``cuda.h`` and
    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``cuda_sources`` per se. To bind
    to a CUDA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    The sources in ``sycl_sources`` are concatenated into a separate ``.sycl``
    file and  prepended with ``torch/types.h``, ``sycl/sycl.hpp`` includes.
    The ``.cpp`` and ``.sycl`` files are compiled separately, but ultimately
    linked into a single library. Note that no bindings are generated for
    functions in ``sycl_sources`` per se. To bind to a SYCL kernel, you must
    create a C++ function that calls it, and either declare or define this
    C++ function in one of the ``cpp_sources`` (and include its name
    in ``functions``).



    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        sycl_sources: A string, or list of strings, containing SYCL source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``sycl_sources`` is
            provided. Set it to ``True`` to force SYCL headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.
        no_implicit_headers: If ``True``, skips automatically adding headers, most notably
            ``#include <torch/extension.h>`` and ``#include <torch/types.h>`` lines.
            Use this option to improve cold start times when you
            already include the necessary headers in your source code. Default: ``False``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(name='inline_extension',
        ...                      cpp_sources=[source],
        ...                      functions=['sin_add'])

    .. note::
        Since load_inline will just-in-time compile the source code, please ensure
        that you have the right toolchains installed in the runtime. For example,
        when loading C++, make sure a C++ compiler is available. If you're loading
        a CUDA extension, you will need to additionally install the corresponding CUDA
        toolkit (nvcc and any other dependencies your code has). Compiling toolchains
        are not included when you install torch and must be additionally installed.

        During compiling, by default, the Ninja backend uses #CPUS + 2 workers to build
        the extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or _get_build_directory(name, verbose)

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    cuda_sources = cuda_sources or []
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]
    sycl_sources = sycl_sources or []
    if isinstance(sycl_sources, str):
        sycl_sources = [sycl_sources]

    if not no_implicit_headers:
        cpp_sources.insert(0, '#include <torch/extension.h>')

    if use_pch is True:
        # Using PreCompile Header('torch/extension.h') to reduce compile time.
        _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
    else:
        remove_extension_h_precompiler_headers()

    # If `functions` is supplied, we create the pybind11 bindings for the user.
    # Here, `functions` is (or becomes, after some processing) a map from
    # function names to function docstrings.
    if functions is not None:
        module_def = []
        module_def.append('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            # Make the function docstring the same as the function name.
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(f"Expected 'functions' to be a list or dict, but was {type(functions)}")
        for function_name, docstring in functions.items():
            if with_pytorch_error_handling:
                module_def.append(f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");')
            else:
                module_def.append(f'm.def("{function_name}", {function_name}, "{docstring}");')
        module_def.append('}')
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, 'main.cpp')
    _maybe_write(cpp_source_path, "\n".join(cpp_sources))

    sources = [cpp_source_path]

    if cuda_sources:
        if not no_implicit_headers:
            cuda_sources.insert(0, '#include <torch/types.h>')
            cuda_sources.insert(1, '#include <cuda.h>')
            cuda_sources.insert(2, '#include <cuda_runtime.h>')

        cuda_source_path = os.path.join(build_directory, 'cuda.cu')
        _maybe_write(cuda_source_path, "\n".join(cuda_sources))

        sources.append(cuda_source_path)

    if sycl_sources:
        if not no_implicit_headers:
            sycl_sources.insert(0, '#include <torch/types.h>')
            sycl_sources.insert(1, '#include <sycl/sycl.hpp>')

        sycl_source_path = os.path.join(build_directory, 'sycl.sycl')
        _maybe_write(sycl_source_path, "\n".join(sycl_sources))

        sources.append(sycl_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates)


def _jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_sycl_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: bool | None,
                 with_sycl: bool | None,
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> types.ModuleType | str:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any('cudnn' in f for f in extra_ldflags or [])
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_directory=build_directory,
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )
    if version > 0:
        if version != old_version and verbose:
            logger.info('The input conditions for extension module %s have changed.', name)
            logger.info('Bumping to version %s and re-building as %s_v%s...', version, name, version)
        name = f'{name}_v{version}'

    baton = FileBaton(os.path.join(build_directory, 'lock'))
    if baton.try_acquire():
        try:
            if version != old_version:
                from .hipify import hipify_python
                from .hipify.hipify_python import GeneratedFileCleaner
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        hipify_result = hipify_python.hipify(
                            project_directory=build_directory,
                            output_directory=build_directory,
                            header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                            extra_files=[os.path.abspath(s) for s in sources],
                            ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                            show_detailed=verbose,
                            show_progress=verbose,
                            is_pytorch_extension=True,
                            clean_ctx=clean_ctx
                        )

                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            hipified_sources.add(hipify_result[s_abs].hipified_path if s_abs in hipify_result else s_abs)

                        sources = list(hipified_sources)

                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_cuda_cflags=extra_cuda_cflags or [],
                        extra_sycl_cflags=extra_sycl_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_cuda=with_cuda,
                        with_sycl=with_sycl,
                        is_standalone=is_standalone)
            elif verbose:
                logger.debug('No modifications detected for re-loaded extension module %s, skipping build step...', name)
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        logger.info('Loading extension module %s...', name)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)

def _get_hipcc_path():
    if IS_WINDOWS:
        # mypy thinks ROCM_VERSION is None but it will never be None here
        hipcc_exe = 'hipcc.exe' if ROCM_VERSION >= (6, 4) else 'hipcc.bat'  # type: ignore[operator]
        return _join_rocm_home('bin', hipcc_exe)
    else:
        return _join_rocm_home('bin', 'hipcc')

def _write_ninja_file_and_compile_objects(
        sources: list[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        sycl_cflags,
        sycl_post_cflags,
        sycl_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: bool | None,
        with_sycl: bool | None) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        logger.debug('Emitting ninja build file %s...', build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=sycl_post_cflags,
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda,
        with_sycl=with_sycl)
    if verbose:
        logger.info('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix='Error compiling objects for extension')


def _write_ninja_file_and_build_library(
        name,
        sources: list[str],
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory: str,
        verbose: bool,
        with_cuda: bool | None,
        with_sycl: bool | None,
        is_standalone: bool = False) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [],
        with_cuda,
        with_sycl,
        verbose,
        is_standalone)
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        logger.debug('Emitting ninja build file %s...', build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
        extra_sycl_cflags=extra_sycl_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_standalone=is_standalone)

    if verbose:
        logger.info('Building extension module %s...', name)
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix=f"Error building extension '{name}'")


def is_ninja_available() -> bool:
    """Return ``True`` if the `ninja <https://ninja-build.org/>`_ build system is available on the system, ``False`` otherwise."""
    try:
        subprocess.check_output(['ninja', '--version'])
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability() -> None:
    """Raise ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not available on the system, does nothing otherwise."""
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions (pip install ninja to get it)")


def _prepare_ldflags(extra_ldflags, with_cuda, with_sycl, verbose, is_standalone):
    if IS_WINDOWS:
        python_lib_path = os.path.join(sys.base_exec_prefix, 'libs')

        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_hip.lib' if IS_HIP_EXTENSION else 'c10_cuda.lib')
        if with_sycl:
            extra_ldflags.append('c10_xpu.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_hip.lib' if IS_HIP_EXTENSION else 'torch_cuda.lib')
            # /INCLUDE is used to ensure torch_cuda is linked against in a project that relies on it.
            # Related issue: https://github.com/pytorch/pytorch/issues/31611
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
        if with_sycl:
            extra_ldflags.append('torch_xpu.lib')
        extra_ldflags.append('torch.lib')
        extra_ldflags.append(f'/LIBPATH:{TORCH_LIB_PATH}')
        if not is_standalone:
            extra_ldflags.append('torch_python.lib')
            extra_ldflags.append(f'/LIBPATH:{python_lib_path}')

    else:
        extra_ldflags.append(f'-L{TORCH_LIB_PATH}')
        extra_ldflags.append('-lc10')
        if with_cuda:
            extra_ldflags.append('-lc10_hip' if IS_HIP_EXTENSION else '-lc10_cuda')
        if with_sycl:
            extra_ldflags.append('-lc10_xpu')
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        if with_sycl:
            extra_ldflags.append('-ltorch_xpu')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')

        if is_standalone:
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    if with_cuda:
        if verbose:
            logger.info('Detected CUDA files, patching ldflags')
        if IS_WINDOWS and not IS_HIP_EXTENSION:
            extra_ldflags.append(f'/LIBPATH:{_join_cuda_home("lib", "x64")}')
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'/LIBPATH:{os.path.join(CUDNN_HOME, "lib", "x64")}')
        elif not IS_HIP_EXTENSION:
            extra_lib_dir = "lib64"
            if (not os.path.exists(_join_cuda_home(extra_lib_dir)) and
                    os.path.exists(_join_cuda_home("lib"))):
                # 64-bit CUDA may be installed in "lib"
                # Note that it's also possible both don't exist (see _find_cuda_home) - in that case we stay with "lib64"
                extra_lib_dir = "lib"
            extra_ldflags.append(f'-L{_join_cuda_home(extra_lib_dir)}')
            extra_ldflags.append('-lcudart')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'-L{os.path.join(CUDNN_HOME, "lib64")}')
        elif IS_HIP_EXTENSION:
            if IS_WINDOWS:
                extra_ldflags.append(f'/LIBPATH:{_join_rocm_home("lib")}')
                extra_ldflags.append('amdhip64.lib')
            else:
                extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
                extra_ldflags.append('-lamdhip64')
    if with_sycl:
        if IS_WINDOWS:
            extra_ldflags.append(f'/LIBPATH:{_join_sycl_home("lib")}')
            extra_ldflags.append('sycl.lib')
        else:
            extra_ldflags.append(f'-L{_join_sycl_home("lib")}')
            extra_ldflags.append('-lsycl')
    return extra_ldflags


def _get_cuda_arch_flags(cflags: list[str] | None = None) -> list[str]:
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            if 'arch' in flag:
                return []

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta+Tegra', '7.2'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere+Tegra', '8.7'),
        ('Ampere', '8.0;8.6+PTX'),
        ('Ada', '8.9+PTX'),
        ('Hopper', '9.0+PTX'),
        ('Blackwell+Tegra', '11.0'),
        ('Blackwell', '10.0;10.3;12.0;12.1+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a',
                        '10.0', '10.0a', '11.0', '11.0a', '10.3', '10.3a', '12.0',
                        '12.0a', '12.1', '12.1a']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given or set as native, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list or _arch_list == "native":
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int("".join(re.findall(r"\d+", arch.split('_')[1])))
                            for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            # Capability of the device may be higher than what's supported by the user's
            # NVCC, causing compilation error. User's NVCC is expected to match the one
            # used to build pytorch, so we use the maximum supported capability of pytorch
            # to clamp the capability.
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += '+PTX'

        if not _arch_list:
            # Only log on rank 0 in distributed settings to avoid spam
            if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                arch_list_str = ';'.join(arch_list)
                logger.debug(
                    "TORCH_CUDA_ARCH_LIST is not set, using TORCH_CUDA_ARCH_LIST='%s' "
                    "for visible GPU architectures. Set os.environ['TORCH_CUDA_ARCH_LIST'] to override.",
                    arch_list_str)
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archival in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archival)

        arch_list = _arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            # Handle both single and double-digit architecture versions
            version = arch.split('+')[0]  # Remove "+PTX" if present
            major, minor = version.split('.')
            num = f"{major}{minor}"
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(set(flags))


def _get_rocm_arch_flags(cflags: list[str] | None = None) -> list[str]:
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`). If user also specified -fgpu-rdc or -fno-gpu-rdc, we
    # assume they know what they're doing. Otherwise, we force -fno-gpu-rdc default.
    has_gpu_rdc_flag = False
    if cflags is not None:
        has_custom_flags = False
        for flag in cflags:
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                has_custom_flags = True
            elif 'gpu-rdc' in flag:
                has_gpu_rdc_flag = True
        if has_custom_flags:
            return [] if has_gpu_rdc_flag else ['-fno-gpu-rdc']
    # Use same defaults as used for building PyTorch
    # Allow env var to override, just like during initial cmake build.
    _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
    if not _archs:
        archFlags = torch._C._cuda_getArchFlags()
        if archFlags:
            archs = archFlags.split()
        else:
            archs = []
            logger.warning(
                "Failed to auto-detect ROCm architecture. Extensions will be compiled "
                "without architecture-specific optimizations. Set PYTORCH_ROCM_ARCH "
                "environment variable to specify target architectures "
                "(e.g., export PYTORCH_ROCM_ARCH='gfx90a;gfx942')."
            )
    else:
        archs = _archs.replace(' ', ';').split(';')
    flags = [f'--offload-arch={arch}' for arch in archs]
    flags += [] if has_gpu_rdc_flag else ['-fno-gpu-rdc']
    return flags

def _get_build_directory(name: str, verbose: bool) -> str:
    """
    Get the build directory for an extension.

    Args:
        name: The name of the extension
        verbose: Whether to print verbose information

    Returns:
        The path to the build directory
    """
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        # Determine GPU accelerator prefix based on available accelerators. Fallback to CPU.
        # Priority: ROCm/HIP > CUDA > CPU
        # Note: torch.backends.cuda.is_built() returns True for both CUDA and ROCm,
        # so we need to check torch.version.hip to distinguish them
        if torch.version.hip is not None:
            accelerator_str = f'rocm{torch.version.hip.replace(".", "")}'
        elif torch.version.cuda is not None:
            accelerator_str = f'cu{torch.version.cuda.replace(".", "")}'
        else:
            accelerator_str = 'cpu'
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
        build_folder = f'{python_version}_{accelerator_str}'

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder)

    if verbose:
        logger.info('Using %s as PyTorch extensions root...', root_extensions_directory)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating extension directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


def _get_num_workers(verbose: bool) -> int | None:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            logger.debug('Using envvar MAX_JOBS (%s) as the number of workers...', max_jobs)
        return int(max_jobs)
    if verbose:
        logger.info(
            'Allowing ninja to set a default number of workers... '
            '(overridable by setting the environment variable MAX_JOBS=N)'
        )
    return None


def _get_vc_env(vc_arch: str) -> dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[attr-defined]

        return distutils._msvccompiler._get_vc_env(vc_arch)
    except AttributeError:
        try:
            from setuptools._distutils import _msvccompiler
            return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[attr-defined]
        except AttributeError:
            from setuptools._distutils.compilers.C import msvc
            return msvc._get_vc_env(vc_arch)  # type: ignore[attr-defined]

def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils  # type: ignore[attr-defined]

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]
        vc_env = {k.upper(): v for k, v in _get_vc_env(plat_spec).items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detaches __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            shell=IS_WINDOWS and IS_HIP_EXTENSION,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env)
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `output`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _get_exec_path(module_name, path):
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv('PATH', '').split(';'):
        torch_lib_in_path = any(
            os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH)
            for p in os.getenv('PATH', '').split(';')
        )
        if not torch_lib_in_path:
            os.environ['PATH'] = f"{TORCH_LIB_PATH};{os.getenv('PATH', '')}"
    return os.path.join(path, f'{module_name}{EXEC_EXT}')


def _import_module_from_library(module_name, path, is_python_module):
    filepath = os.path.join(path, f"{module_name}{LIB_EXT}")
    if is_python_module:
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            raise AssertionError(f"Failed to create spec for module {module_name} at {filepath}")
        module = importlib.util.module_from_spec(spec)
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise AssertionError("spec.loader is not a valid importlib Loader")
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)
        return filepath


def _write_ninja_file_to_build_library(path,
                                       name,
                                       sources,
                                       extra_cflags,
                                       extra_cuda_cflags,
                                       extra_sycl_cflags,
                                       extra_ldflags,
                                       extra_include_paths,
                                       with_cuda,
                                       with_sycl,
                                       is_standalone) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_sycl_cflags = [flag.strip() for flag in extra_sycl_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    # TODO generalize with_cuda as specific device type.
    if with_cuda:
        system_includes = include_paths("cuda")
    elif with_sycl:
        system_includes = include_paths("xpu")
    else:
        system_includes = include_paths("cpu")
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')

    # Windows does not understand `-isystem` and quotes flags later.
    if IS_WINDOWS:
        common_cflags += [f'-I{include}' for include in user_includes + system_includes]
    else:
        common_cflags += [f'-I{shlex.quote(include)}' for include in user_includes]
        common_cflags += [f'-isystem {shlex.quote(include)}' for include in system_includes]

    if IS_WINDOWS:
        COMMON_HIP_FLAGS.extend(['-fms-runtime-lib=dll'])
        cflags = common_cflags + ['/std:c++17'] + extra_cflags
        cflags += COMMON_MSVC_FLAGS + (COMMON_HIP_FLAGS if IS_HIP_EXTENSION else [])
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ['-fPIC', '-std=c++17'] + extra_cflags

    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = ['-DWITH_HIP'] + common_cflags + extra_cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags = cuda_flags + ['-std=c++17']
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
        cuda_flags += extra_cuda_cflags
        if IS_WINDOWS:
            cuda_flags = _nt_quote_args(cuda_flags)
    elif with_cuda:
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags(extra_cuda_cflags)
        if IS_WINDOWS:
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ['-Xcompiler', flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cuda_flags
            cuda_flags = cuda_flags + ['-std=c++17']
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any(flag.startswith('-std=') for flag in cuda_flags):
                cuda_flags.append('-std=c++17')
            cc_env = os.getenv("CC")
            if cc_env is not None:
                cuda_flags = ['-ccbin', cc_env] + cuda_flags
    else:
        cuda_flags = None

    if with_sycl:
        sycl_cflags = cflags + _COMMON_SYCL_FLAGS
        sycl_cflags += extra_sycl_cflags
        _append_sycl_targets_if_missing(sycl_cflags)
        _append_sycl_std_if_no_std_present(sycl_cflags)
        host_cflags = cflags
        # escaping quoted arguments to pass them thru SYCL compiler
        icpx_version = _get_icpx_version()
        if int(icpx_version) < 20250200:
            host_cflags = [item.replace('\\"', '\\\\"') for item in host_cflags]

        sycl_cflags += _wrap_sycl_host_flags(host_cflags)
        sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
        sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_cflags)
    else:
        sycl_cflags = None
        sycl_dlink_post_cflags = None

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            # Use a different object filename in case a C++ and CUDA file have
            # the same filename but different extension (.cpp vs. .cu).
            target = f'{file_name}.cuda.o'
        elif _is_sycl_file(source_file) and with_sycl:
            target = f'{file_name}.sycl.o'
        else:
            target = f'{file_name}.o'
        return target

    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags

    # The darwin linker needs explicit consent to ignore unresolved symbols.
    if IS_MACOS:
        ldflags.append('-undefined dynamic_lookup')
    elif IS_WINDOWS:
        ldflags = _nt_quote_args(ldflags)

    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f'{name}{ext}'

    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        cuda_cflags=cuda_flags,
        cuda_post_cflags=None,
        cuda_dlink_post_cflags=None,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=[],
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_cuda=with_cuda,
        with_sycl=with_sycl)


def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sycl_cflags,
                      sycl_post_cflags,
                      sycl_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      with_sycl) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_post_cflags`: list of flags to append to the $nvcc invocation. Can be None.
    `cuda_dlink_post_cflags`: list of flags to append to the $nvcc device code link invocation. Can be None.
    `sycl_cflags`: list of flags to pass to SYCL compiler. Can be None.
    `sycl_post_cflags`: list of flags to append to the SYCL compiler invocation. Can be None.
    `sycl_dlink_post_cflags`: list of flags to append to the SYCL compiler device code link invocation. Can be None.
e.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    sycl_cflags = sanitize_flags(sycl_cflags)
    sycl_post_cflags = sanitize_flags(sycl_post_cflags)
    sycl_dlink_post_cflags = sanitize_flags(sycl_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    if len(sources) != len(objects):
        raise AssertionError("sources and objects lists must be the same length")
    if len(sources) == 0:
        raise AssertionError("At least one source is required to build a library")

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            if IS_HIP_EXTENSION:
                nvcc = _get_hipcc_path()
            else:
                nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc = {nvcc}')
    if with_sycl or sycl_dlink_post_cflags:
        sycl = 'icx' if IS_WINDOWS else 'icpx'
        config.append(f'sycl = {sycl}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    if with_sycl:
        flags.append(f'sycl_cflags = {" ".join(sycl_cflags)}')
        flags.append(f'sycl_post_cflags = {" ".join(sycl_post_cflags)}')
    flags.append(f'sycl_dlink_post_cflags = {" ".join(sycl_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compiler_name = "$cxx" if IS_HIP_EXTENSION else "cl"
        compile_rule.append(
            f'  command = {compiler_name} '
            '/showIncludes $cflags -c $in /Fo$out $post_cflags'  # codespell:ignore
        )
        if not IS_HIP_EXTENSION:
            compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # -MD is not supported by ROCm
        # Nvcc flag `-MD` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use -MD to make this work on Windows too.
            nvcc_gendeps = '-MD -MF $out.d'
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    if with_sycl:
        sycl_compile_rule = ['rule sycl_compile']
        # SYCL compiler does not recognize .sycl extension automatically,
        # so we pass '-x c++' explicitly notifying compiler of file format
        sycl_compile_rule.append(
            '  command = $sycl $sycl_cflags -c -x c++ $in -o $out $sycl_post_cflags')


    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects, strict=True):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        is_sycl_source = _is_sycl_file(source_file) and with_sycl
        if is_cuda_source:
            rule = 'cuda_compile'
        elif is_sycl_source:
            rule = 'sycl_compile'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        cuda_devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        cuda_devlink_rule = ['rule cuda_devlink']
        cuda_devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        cuda_devlink = [f'build {cuda_devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [cuda_devlink_out]
    else:
        cuda_devlink_rule, cuda_devlink = [], []

    if sycl_dlink_post_cflags:
        sycl_devlink_out = os.path.join(os.path.dirname(objects[0]), "sycl_dlink.o")
        if IS_WINDOWS:
            sycl_devlink_objects = [obj.replace(":", "$:") for obj in objects]
            objects += [sycl_devlink_out]
            sycl_devlink_out = sycl_devlink_out.replace(":", "$:")
        else:
            sycl_devlink_objects = list(objects)
            objects += [sycl_devlink_out]
        sycl_devlink_rule = ["rule sycl_devlink"]
        sycl_devlink_rule.append(
            "  command = $sycl $in -o $out $sycl_dlink_post_cflags"
        )
        sycl_devlink = [
            f"build {sycl_devlink_out}: sycl_devlink {' '.join(sycl_devlink_objects)}"
        ]
    else:
        sycl_devlink_rule, sycl_devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
    if with_sycl:
        blocks.append(sycl_compile_rule)  # type: ignore[possibly-undefined]
    blocks += [cuda_devlink_rule, sycl_devlink_rule, link_rule, build, cuda_devlink, sycl_devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)

def _join_cuda_home(*paths) -> str:
    """
    Join paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    """
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME environment variable is not set. '
                      'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext

def _is_sycl_file(path: str) -> bool:
    valid_ext = ['.sycl']
    return os.path.splitext(path)[1] in valid_ext
