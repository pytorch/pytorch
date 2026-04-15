# mypy: allow-untyped-defs
"""C++/CUDA/SYCL extension utilities.

Historically a single ``torch/utils/cpp_extension.py`` module. The module is
in the process of being split into focused submodules; for now all
implementation lives in :mod:`torch.utils.cpp_extension._impl` and is
re-exported here to preserve the full public API.
"""

from ._impl import *  # noqa: F401, F403
from ._impl import __all__  # noqa: F401

# Re-export names not in _impl.__all__ that are consumed elsewhere in the
# PyTorch source tree or by downstream projects. Keep this list in sync as
# implementation is moved into the dedicated submodules.
from ._impl import (  # noqa: F401
    ABI_INCOMPATIBILITY_WARNING,
    BUILT_FROM_SOURCE_VERSION_PATTERN,
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
    HIP_HOME,
    IS_HIP_EXTENSION,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    JIT_EXTENSION_VERSIONER,
    LIB_EXT,
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
    VersionMap,
    VersionRange,
    WINDOWS_CUDA_HOME,
    WRONG_COMPILER_WARNING,
    _COMMON_SYCL_FLAGS,
    _find_cuda_home,
    _find_rocm_home,
    _find_sycl_home,
    _join_rocm_home,
    _SYCL_DLINK_FLAGS,
    _TORCH_PATH,
)


def __getattr__(name):
    # The historical ``torch/utils/cpp_extension.py`` module exposed all of its
    # module-level names (including those starting with underscore) via plain
    # attribute access. Now that it's a package, preserve that surface by
    # transparently forwarding unknown attributes to the implementation module.
    from . import _impl

    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(
            f"module 'torch.utils.cpp_extension' has no attribute {name!r}"
        ) from None
