import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections

import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
from typing import Dict, List, Optional, Union, Tuple
from torch.torch_version import TorchVersion

from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore[attr-defined]

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

VersionRange = Tuple[Tuple[int, ...], Tuple[int, ...]]
VersionMap = Dict[str, VersionRange]
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
}

__all__ = ["get_default_build_root", "check_compiler_ok_for_platform", "get_compiler_abi_compatibility_and_version", "BuildExtension",
           "CppExtension", "CUDAExtension", "include_paths", "library_paths", "load", "load_inline", "is_ninja_available",
           "verify_ninja_availability"]
# Taken directly from python stdlib < 3.9
# See https://github.com/pytorch/pytorch/issues/48617
def _nt_quote_args(args: Optional[List[str]]) -> List[str]:
    """Quote command-line arguments for DOS/Windows conventions.

    Just wraps every argument which contains blanks in double quotes, and
    returns a new argument list.
    """
    # Cover None-type
    if not args:
        return []
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]

def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
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
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'",
              file=sys.stderr)
    return cuda_home

def _find_rocm_home() -> Optional[str]:
    r'''Finds the ROCm install path.'''
    # Guess #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Guess #2
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            # this will be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            # Guess #3
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'",
              file=sys.stderr)
    return rocm_home


def _join_rocm_home(*paths) -> str:
    r'''
    Joins paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    '''
    if ROCM_HOME is None:
        raise EnvironmentError('ROCM_HOME environment variable is not set. '
                               'Please set it to your ROCm install root.')
    elif IS_WINDOWS:
        raise EnvironmentError('Building PyTorch extensions using '
                               'ROCm and Windows is not supported.')
    return os.path.join(ROCM_HOME, *paths)


ABI_INCOMPATIBILITY_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 5.0 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 5 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
WRONG_COMPILER_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({user_compiler}) is not compatible with the compiler Pytorch was
built with for this platform, which is {pytorch_compiler} on {platform}. Please
use {pytorch_compiler} to to compile your extension. Alternatively, you may
compile PyTorch from source using {user_compiler}, and then you can also use
{user_compiler} to compile your extension.

See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
with compiling PyTorch from source.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
CUDA_MISMATCH_MESSAGE = '''
The detected CUDA version ({0}) mismatches the version that was used to compile
PyTorch ({1}). Please make sure to use the same CUDA versions.
'''
CUDA_MISMATCH_WARN = "The detected CUDA version ({0}) has a minor version mismatch with the version that was used to compile PyTorch ({1}). Most likely this shouldn't be a problem."
CUDA_NOT_FOUND_MESSAGE = '''
CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH
environment variable or add NVCC to your system PATH. The extension compilation will fail.
'''
ROCM_HOME = _find_rocm_home()
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

CUDA_HOME = _find_cuda_home()
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
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
    '-fPIC',
    '-D__HIP_PLATFORM_HCC__=1',
    '-DUSE_ROCM=1',
]

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
]

JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}


def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> List[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']


def get_default_build_root() -> str:
    r'''
    Returns the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    '''
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


def check_compiler_ok_for_platform(compiler: str) -> bool:
    r'''
    Verifies that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    '''
    if IS_WINDOWS:
        return True
    which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            return version_string.startswith('clang version')
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def get_compiler_abi_compatibility_and_version(compiler) -> Tuple[bool, TorchVersion]:
    r'''
    Determine if the given compiler is ABI-compatible with PyTorch alongside
    its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    '''
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        warnings.warn(WRONG_COMPILER_WARNING.format(
            user_compiler=compiler,
            pytorch_compiler=_accepted_compilers_for_platform()[0],
            platform=sys.platform))
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
        return (False, TorchVersion('0.0.0'))

    if tuple(map(int, version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(version)))

    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))

    return (False, TorchVersion('.'.join(version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
    if cuda_version is None:
        return

    cuda_str_version = cuda_version.group(1)
    cuda_ver = packaging.version.parse(cuda_str_version)
    torch_cuda_version = packaging.version.parse(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
        warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))

    if not (sys.platform.startswith('linux') and
            os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and
            _is_binary_build()):
        return

    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS

    if cuda_str_version not in cuda_compiler_bounds:
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
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


# See below for why we inherit BuildExtension from object.
# https://stackoverflow.com/questions/1713038/super-fails-with-error-typeerror-argument-1-must-be-type-not-classobj-when


class BuildExtension(build_ext):
    r'''
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''

    @classmethod
    def with_options(cls, **options):
        r'''
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
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
                   '{}. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        if cuda_ext and not IS_HIP_EXTENSION:
            _check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'nvcc' is
            # passed to extra_compile_args in CUDAExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

            if 'nvcc_dlink' in extension.extra_compile_args:
                assert self.use_ninja, f"With dlink=True, ninja is required to build cuda extension {extension.name}."

        # Register .cu, .cuh, .hip, and .mm as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']
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

        def convert_to_absolute_paths_inplace(paths):
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
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
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
            else:
                cuda_dlink_post_cflags = None
            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (COMMON_NVCC_FLAGS +
                    cflags + _get_cuda_arch_flags(cflags))

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

                obj_regex = re.compile('/Fo(.*)')
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
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        cflags = win_cuda_flags(cflags) + ['-std=c++17', '--use-local-env']
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                            cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
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
                                   depends=None):

            if not self.compiler.initialized:
                self.compiler.initialize()
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any,
            # For ninja build, the build location is not local, the build happens
            # in a in script created build folder, relative path lost their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            common_cflags.extend(COMMON_MSVC_FLAGS)
            cflags = cflags + common_cflags + pp_opts
            with_cuda = any(map(_is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['-std=c++17', '--use-local-env']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                    cuda_cflags.append('-Xcudafe')
                    cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
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

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511
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
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self) -> Tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                   'This may lead to multiple activations of the VC env.'
                   'Please set `DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))


def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

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
        ...             extra_compile_args=['-g']),
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    '''
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
    libraries.append('torch_python')
    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

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
        ...                                     'nvcc': ['-O2']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, Pytorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py

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
    help reduce the protentional perf degradation of `-rdc`.
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
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        assert ROCM_VERSION is not None
        libraries.append('amdhip64' if ROCM_VERSION >= (3, 5) else 'hip_hcc')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    if IS_HIP_EXTENSION:
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
            hipified_s_abs = (hipify_result[s_abs]["hipified_path"] if (s_abs in hipify_result and
                              hipify_result[s_abs]["hipified_path"] is not None) else s_abs)
            # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
            # *never* absolute paths
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

        sources = list(hipified_sources)

    include_dirs += include_paths(cuda=True)
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

        if (torch.version.cuda is not None) and packaging.version.parse(torch.version.cuda) >= packaging.version.parse('11.2'):
            extra_compile_args_dlink += ['-dlto']   # Device Link Time Optimization started from cuda 11.2

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


def include_paths(cuda: bool = False) -> List[str]:
    '''
    Get the include paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific include paths.

    Returns:
        A list of include path strings.
    '''
    lib_include = os.path.join(_TORCH_PATH, 'include')
    paths = [
        lib_include,
        # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
        os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        # Some internal (old) Torch headers don't properly prefix their includes,
        # so we need to pass -Itorch/lib/include/TH as well.
        os.path.join(lib_include, 'TH'),
        os.path.join(lib_include, 'THC')
    ]
    if cuda and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    elif cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    return paths


def library_paths(cuda: bool = False) -> List[str]:
    r'''
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific library paths.

    Returns:
        A list of library path strings.
    '''
    # We need to link against libtorch.so
    paths = [TORCH_LIB_PATH]

    if cuda and IS_HIP_EXTENSION:
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    elif cuda:
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
    return paths


def load(name,
         sources: Union[str, List[str]],
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False,
         with_cuda: Optional[bool] = None,
         is_python_module=True,
         is_standalone=False,
         keep_intermediates=True):
    r'''
    Loads a PyTorch C++ extension just-in-time (JIT).

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

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
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
    '''
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory or _get_build_directory(name, verbose),
        verbose,
        with_cuda,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates)


def load_inline(name,
                cpp_sources,
                cuda_sources=None,
                functions=None,
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_ldflags=None,
                extra_include_paths=None,
                build_directory=None,
                verbose=False,
                with_cuda=None,
                is_python_module=True,
                with_pytorch_error_handling=True,
                keep_intermediates=True):
    r'''
    Loads a PyTorch C++ extension just-in-time (JIT) from string sources.

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
    <torch/extension.h>``.

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``
    file and  prepended with ``torch/types.h``, ``cuda.h`` and
    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``cuda_sources`` per  se. To bind
    to a CUDA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.

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
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or _get_build_directory(name, verbose)

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    cuda_sources = cuda_sources or []
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]

    cpp_sources.insert(0, '#include <torch/extension.h>')

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
                module_def.append(
                    'm.def("{0}", torch::wrap_pybind_function({0}), "{1}");'
                    .format(function_name, docstring))
            else:
                module_def.append('m.def("{0}", {0}, "{1}");'.format(function_name, docstring))
        module_def.append('}')
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, 'main.cpp')
    with open(cpp_source_path, 'w') as cpp_source_file:
        cpp_source_file.write('\n'.join(cpp_sources))

    sources = [cpp_source_path]

    if cuda_sources:
        cuda_sources.insert(0, '#include <torch/types.h>')
        cuda_sources.insert(1, '#include <cuda.h>')
        cuda_sources.insert(2, '#include <cuda_runtime.h>')

        cuda_source_path = os.path.join(build_directory, 'cuda.cu')
        with open(cuda_source_path, 'w') as cuda_source_file:
            cuda_source_file.write('\n'.join(cuda_sources))

        sources.append(cuda_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_cuda,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates)


def _jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: Optional[bool],
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> None:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any('cudnn' in f for f in extra_ldflags or [])
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_directory=build_directory,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )
    if version > 0:
        if version != old_version and verbose:
            print(f'The input conditions for extension module {name} have changed. ' +
                  f'Bumping to version {version} and re-building as {name}_v{version}...',
                  file=sys.stderr)
        name = f'{name}_v{version}'

    if version != old_version:
        baton = FileBaton(os.path.join(build_directory, 'lock'))
        if baton.try_acquire():
            try:
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
                            hipified_sources.add(hipify_result[s_abs]["hipified_path"] if s_abs in hipify_result else s_abs)

                        sources = list(hipified_sources)

                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_cuda_cflags=extra_cuda_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_cuda=with_cuda,
                        is_standalone=is_standalone)
            finally:
                baton.release()
        else:
            baton.wait()
    elif verbose:
        print('No modifications detected for re-loaded extension '
              f'module {name}, skipping build step...',
              file=sys.stderr)

    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)


def _write_ninja_file_and_compile_objects(
        sources: List[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool]) -> None:
    verify_ninja_availability()
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda)
    if verbose:
        print('Compiling objects...', file=sys.stderr)
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix='Error compiling objects for extension')


def _write_ninja_file_and_build_library(
        name,
        sources: List[str],
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool],
        is_standalone: bool = False) -> None:
    verify_ninja_availability()
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [],
        with_cuda,
        verbose,
        is_standalone)
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)
    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_cuda=with_cuda,
        is_standalone=is_standalone)

    if verbose:
        print(f'Building extension module {name}...', file=sys.stderr)
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix=f"Error building extension '{name}'")


def is_ninja_available():
    r'''
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    '''
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    r'''
    Raises ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not
    available on the system, does nothing otherwise.
    '''
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")


def _prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
    if IS_WINDOWS:
        python_path = os.path.dirname(sys.executable)
        python_lib_path = os.path.join(python_path, 'libs')

        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_cuda.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_cuda.lib')
            # /INCLUDE is used to ensure torch_cuda is linked against in a project that relies on it.
            # Related issue: https://github.com/pytorch/pytorch/issues/31611
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
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
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')

        if is_standalone and "TBB" in torch.__config__.parallel_info():
            extra_ldflags.append('-ltbb')

        if is_standalone:
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    if with_cuda:
        if verbose:
            print('Detected CUDA files, patching ldflags', file=sys.stderr)
        if IS_WINDOWS:
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
            assert ROCM_VERSION is not None
            extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
            extra_ldflags.append('-lamdhip64' if ROCM_VERSION >= (3, 5) else '-lhip_hcc')
    return extra_ldflags


def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    r'''
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    '''
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
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
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list:
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int(arch.split('_')[1])
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
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)

        arch_list = _arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            num = arch[0] + arch[2]
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(set(flags))


def _get_rocm_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                return ['-fno-gpu-rdc']
    # Use same defaults as used for building PyTorch
    # Allow env var to override, just like during initial cmake build.
    _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
    if not _archs:
        archFlags = torch._C._cuda_getArchFlags()
        if archFlags:
            archs = archFlags.split()
        else:
            archs = []
    else:
        archs = _archs.replace(' ', ';').split(';')
    flags = ['--offload-arch=%s' % arch for arch in archs]
    flags += ['-fno-gpu-rdc']
    return flags

def _get_build_directory(name: str, verbose: bool) -> str:
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        cu_str = ('cpu' if torch.version.cuda is None else
                  f'cu{torch.version.cuda.replace(".", "")}')  # type: ignore[attr-defined]
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}'
        build_folder = f'{python_version}_{cu_str}'

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder)

    if verbose:
        print(f'Using {root_extensions_directory} as PyTorch extensions root...', file=sys.stderr)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print(f'Creating extension directory {build_directory}...', file=sys.stderr)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...',
                  file=sys.stderr)
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... '
              '(overridable by setting the environment variable MAX_JOBS=N)',
              file=sys.stderr)
    return None


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]

        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
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
        # 1) If the stdout encoding is not utf-8, setuptools detachs __stdout__.
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
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)


def _write_ninja_file_to_build_library(path,
                                       name,
                                       sources,
                                       extra_cflags,
                                       extra_cuda_cflags,
                                       extra_ldflags,
                                       extra_include_paths,
                                       with_cuda,
                                       is_standalone) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    system_includes = include_paths(with_cuda)
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)

    # Windows does not understand `-isystem`.
    if IS_WINDOWS:
        user_includes += system_includes
        system_includes.clear()

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')

    # Note [Pybind11 ABI constants]
    #
    # Pybind11 before 2.4 used to build an ABI strings using the following pattern:
    # f"__pybind11_internals_v{PYBIND11_INTERNALS_VERSION}{PYBIND11_INTERNALS_KIND}{PYBIND11_BUILD_TYPE}__"
    # Since 2.4 compier type, stdlib and build abi parameters are also encoded like this:
    # f"__pybind11_internals_v{PYBIND11_INTERNALS_VERSION}{PYBIND11_INTERNALS_KIND}{PYBIND11_COMPILER_TYPE}{PYBIND11_STDLIB}{PYBIND11_BUILD_ABI}{PYBIND11_BUILD_TYPE}__"
    #
    # This was done in order to further narrow down the chances of compiler ABI incompatibility
    # that can cause a hard to debug segfaults.
    # For PyTorch extensions we want to relax those restrictions and pass compiler, stdlib and abi properties
    # captured during PyTorch native library compilation in torch/csrc/Module.cpp

    for pname in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
        pval = getattr(torch._C, f"_PYBIND11_{pname}")
        if pval is not None and not IS_WINDOWS:
            common_cflags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')

    common_cflags += [f'-I{include}' for include in user_includes]
    common_cflags += [f'-isystem {include}' for include in system_includes]

    common_cflags += ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]

    if IS_WINDOWS:
        cflags = common_cflags + COMMON_MSVC_FLAGS + ['/std:c++17'] + extra_cflags
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ['-fPIC', '-std=c++17'] + extra_cflags

    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = ['-DWITH_HIP'] + cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags += extra_cuda_cflags
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
    elif with_cuda:
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
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

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            # Use a different object filename in case a C++ and CUDA file have
            # the same filename but different extension (.cpp vs. .cu).
            target = f'{file_name}.cuda.o'
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
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_cuda=with_cuda)


def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
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
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            if IS_HIP_EXTENSION:
                nvcc = _join_rocm_home('bin', 'hipcc')
            else:
                nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile was added in CUDA 10.2.
        # Compilation will work on earlier CUDA versions but header file
        # dependencies are not correctly computed.
        required_cuda_version = packaging.version.parse('11.0')
        has_cuda_version = torch.version.cuda is not None
        if has_cuda_version and packaging.version.parse(torch.version.cuda) >= required_cuda_version:
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            if IS_WINDOWS:
                nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

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
        blocks.append(cuda_compile_rule)
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')


def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext
