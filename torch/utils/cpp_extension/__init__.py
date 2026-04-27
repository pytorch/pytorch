# mypy: allow-untyped-defs
"""C++/CUDA/SYCL extension utilities.

The setuptools-dependent adapters (:class:`BuildExtension`,
:func:`CppExtension`, :func:`CUDAExtension`, :func:`SyclExtension`) live
in :mod:`.setuptools` and are resolved lazily on first access, so
``import torch.utils.cpp_extension`` does not pull in the ``setuptools``
library unless one of those adapters is used.
"""

from ._discovery import (  # noqa: F401
    _COMMON_SYCL_FLAGS,
    _find_cuda_home,
    _find_rocm_home,
    _find_sycl_home,
    _join_rocm_home,
    _SYCL_DLINK_FLAGS,
    _TORCH_PATH,
    ABI_INCOMPATIBILITY_WARNING,
    BUILT_FROM_SOURCE_VERSION_PATTERN,
    check_compiler_is_gcc,
    check_compiler_ok_for_platform,
    CLIB_EXT,
    CLIB_PREFIX,
    COMMON_HIP_FLAGS,
    COMMON_HIPCC_FLAGS,
    COMMON_MSVC_FLAGS,
    COMMON_NVCC_FLAGS,
    CUDA_CLANG_VERSIONS,
    CUDA_GCC_VERSIONS,
    CUDA_HOME,
    CUDA_MISMATCH_MESSAGE,
    CUDA_MISMATCH_WARN,
    CUDA_NOT_FOUND_MESSAGE,
    CUDNN_HOME,
    EXEC_EXT,
    get_compiler_abi_compatibility_and_version,
    get_cxx_compiler,
    get_default_build_root,
    HIP_HOME,
    include_paths,
    IS_HIP_EXTENSION,
    IS_LINUX,
    IS_MACOS,
    is_ninja_available,
    IS_WINDOWS,
    JIT_EXTENSION_VERSIONER,
    LIB_EXT,
    library_paths,
    MINIMUM_CLANG_VERSION,
    MINIMUM_GCC_VERSION,
    MINIMUM_MSVC_VERSION,
    MSVC_IGNORE_CUDAFE_WARNINGS,
    PLAT_TO_VCVARS,
    ROCM_HOME,
    ROCM_VERSION,
    SHARED_FLAG,
    SUBPROCESS_DECODE_ARGS,
    SYCL_HOME,
    TORCH_LIB_PATH,
    verify_ninja_availability,
    VersionMap,
    VersionRange,
    WINDOWS_CUDA_HOME,
    WRONG_COMPILER_WARNING,
)
from ._jit import (  # noqa: F401
    _get_cuda_arch_flags,
    _get_hipcc_path,
    _get_rocm_arch_flags,
    _write_ninja_file_and_compile_objects,
    load,
    load_inline,
    remove_extension_h_precompiler_headers,
)


__all__ = [
    "BuildExtension",
    "check_compiler_is_gcc",
    "check_compiler_ok_for_platform",
    "CppExtension",
    "CUDAExtension",
    "get_compiler_abi_compatibility_and_version",
    "get_cxx_compiler",
    "get_default_build_root",
    "include_paths",
    "is_ninja_available",
    "library_paths",
    "load",
    "load_inline",
    "remove_extension_h_precompiler_headers",
    "SyclExtension",
    "verify_ninja_availability",
]


_SETUPTOOLS_LAZY = frozenset(
    ("BuildExtension", "CppExtension", "CUDAExtension", "SyclExtension")
)


def __getattr__(name):
    if name in _SETUPTOOLS_LAZY:
        from . import setuptools as _st

        value = getattr(_st, name)
        globals()[name] = value
        return value
    # Preserve historical attribute access to names that lived on the old
    # monolithic ``cpp_extension.py`` module but aren't explicitly re-exported
    # above (e.g. private helpers consumed elsewhere in the tree).
    for _sub in ("_jit", "_discovery"):
        _mod = __import__(f"{__name__}.{_sub}", fromlist=[name])
        if hasattr(_mod, name):
            value = getattr(_mod, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
