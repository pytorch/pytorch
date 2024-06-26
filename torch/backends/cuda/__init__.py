# mypy: allow-untyped-defs
import contextlib

from typing import Union
from typing_extensions import deprecated

import torch

__all__ = [
    "is_built",
    "cuFFTPlanCacheAttrContextProp",
    "cuFFTPlanCache",
    "cuFFTPlanCacheManager",
    "cuBLASModule",
    "preferred_linalg_library",
    "preferred_blas_library",
    "cufft_plan_cache",
    "matmul",
    "SDPAParams",
    "enable_cudnn_sdp",
    "cudnn_sdp_enabled",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "can_use_flash_attention",
    "can_use_efficient_attention",
    "can_use_cudnn_attention",
    "sdp_kernel",
]


def is_built():
    r"""
    Return whether PyTorch is built with CUDA support.

    Note that this doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run on a machine with working CUDA drivers and devices, we would be able to use it.
    """
    return torch._C._has_cuda


class cuFFTPlanCacheAttrContextProp:
    # Like regular ContextProp, but uses the `.device_index` attribute from the
    # calling object as the first argument to the getter and setter.
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)


class cuFFTPlanCache:
    r"""
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cuFFT plan cache.
    """

    def __init__(self, device_index):
        self.device_index = device_index

    size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_size,
        ".size is a read-only property showing the number of plans currently in the "
        "cache. To change the cache capacity, set cufft_plan_cache.max_size.",
    )

    max_size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_max_size, torch._cufft_set_plan_cache_max_size
    )

    def clear(self):
        return torch._cufft_clear_plan_cache(self.device_index)


class cuFFTPlanCacheManager:
    r"""
    Represent all cuFFT plan caches, return the cuFFTPlanCache for a given device when indexed.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.cuda._utils._get_device_index(device)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"cufft_plan_cache: expected 0 <= device index < {torch.cuda.device_count()}, but got "
                f"device with index {index}"
            )
        if len(self.caches) == 0:
            self.caches.extend(
                cuFFTPlanCache(index) for index in range(torch.cuda.device_count())
            )
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.cuda.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.cuda.current_device()], name, value)
        else:
            return super().__setattr__(name, value)


class cuBLASModule:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch._C._get_cublas_allow_tf32()
        elif name == "allow_fp16_reduced_precision_reduction":
            return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
        elif name == "allow_bf16_reduced_precision_reduction":
            return torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch._C._set_cublas_allow_tf32(value)
        elif name == "allow_fp16_reduced_precision_reduction":
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
        elif name == "allow_bf16_reduced_precision_reduction":
            return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
        raise AttributeError("Unknown attribute " + name)


_LinalgBackends = {
    "default": torch._C._LinalgBackend.Default,
    "cusolver": torch._C._LinalgBackend.Cusolver,
    "magma": torch._C._LinalgBackend.Magma,
}
_LinalgBackends_str = ", ".join(_LinalgBackends.keys())


def preferred_linalg_library(
    backend: Union[None, str, torch._C._LinalgBackend] = None
) -> torch._C._LinalgBackend:
    r"""
    Override the heuristic PyTorch uses to choose between cuSOLVER and MAGMA for CUDA linear algebra operations.

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA linear algebra operation it often uses the cuSOLVER or MAGMA libraries,
    and if both are available it decides which to use with a heuristic.
    This flag (a :class:`str`) allows overriding those heuristics.

    * If `"cusolver"` is set then cuSOLVER will be used wherever possible.
    * If `"magma"` is set then MAGMA will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between
      cuSOLVER and MAGMA if both are available.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_LINALG_PREFER_CUSOLVER=1 to set the preferred library to cuSOLVER
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's heuristic library selection is incorrect
    for your application's inputs.

    Currently supported linalg operators:

    * :func:`torch.linalg.inv`
    * :func:`torch.linalg.inv_ex`
    * :func:`torch.linalg.cholesky`
    * :func:`torch.linalg.cholesky_ex`
    * :func:`torch.cholesky_solve`
    * :func:`torch.cholesky_inverse`
    * :func:`torch.linalg.lu_factor`
    * :func:`torch.linalg.lu`
    * :func:`torch.linalg.lu_solve`
    * :func:`torch.linalg.qr`
    * :func:`torch.linalg.eigh`
    * :func:`torch.linalg.eighvals`
    * :func:`torch.linalg.svd`
    * :func:`torch.linalg.svdvals`
    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _LinalgBackends:
            raise RuntimeError(
                "Unknown input value. " f"Choose from: {_LinalgBackends_str}."
            )
        torch._C._set_linalg_preferred_backend(_LinalgBackends[backend])
    elif isinstance(backend, torch._C._LinalgBackend):
        torch._C._set_linalg_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return torch._C._get_linalg_preferred_backend()


_BlasBackends = {
    "cublas": torch._C._BlasBackend.Cublas,
    "cublaslt": torch._C._BlasBackend.Cublaslt,
    "hipblaslt": torch._C._BlasBackend.Cublaslt,  # alias
}
_BlasBackends_str = ", ".join(_BlasBackends.keys())


def preferred_blas_library(
    backend: Union[None, str, torch._C._BlasBackend] = None
) -> torch._C._BlasBackend:
    r"""
    Override the library PyTorch uses for BLAS operations. Choose between cuBLAS and cuBLASLt.

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA BLAS operation it defaults to cuBLAS even if both cuBLAS and cuBLASLt are available.
    For PyTorch built for ROCm, hipBLAS and hipBLASLt may offer different performance.
    This flag (a :class:`str`) allows overriding which BLAS library to use.

    * If `"cublas"` is set then cuBLAS will be used wherever possible.
    * If `"cublaslt"` is set then cuBLASLt will be used wherever possible.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_BLAS_PREFER_CUBLASLT=1 to set the preferred library to cuBLASLt
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's library selection is incorrect
    for your application's inputs.

    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                "Unknown input value. " f"Choose from: {_BlasBackends_str}."
            )
        torch._C._set_blas_preferred_backend(_BlasBackends[backend])
    elif isinstance(backend, torch._C._BlasBackend):
        torch._C._set_blas_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return torch._C._get_blas_preferred_backend()


from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend

# Set the __module__ attribute
SDPAParams.__module__ = "torch.backends.cuda"
SDPAParams.__name__ = "SDPAParams"


def flash_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
    return torch._C._get_flash_sdp_enabled()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    torch._C._set_sdp_use_flash(enabled)


def mem_efficient_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return torch._C._get_mem_efficient_sdp_enabled()


def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    torch._C._set_sdp_use_mem_efficient(enabled)


def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    torch._C._set_sdp_use_math(enabled)


def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_flash_attention(params, debug)


def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_mem_efficient_attention(params, debug)


def can_use_cudnn_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if cudnn_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why cuDNN attention could not be run.
            Defaults to False.

    Returns:
        True if cuDNN can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_cudnn_attention(params, debug)


def cudnn_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """
    return torch._C._get_cudnn_sdp_enabled()


def enable_cudnn_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
    torch._C._set_sdp_use_cudnn(enabled)


@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    from torch.nn.attention import sdpa_kernel

    backend_list = []
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    if enable_cudnn:
        backend_list.append(SDPBackend.CUDNN_ATTENTION)

    with sdpa_kernel(backend_list) as context:
        try:
            yield context
        finally:
            pass


cufft_plan_cache = cuFFTPlanCacheManager()
matmul = cuBLASModule()
