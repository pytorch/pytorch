# mypy: allow-untyped-defs
""" This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention """
import contextlib
from typing import List, Union
from warnings import warn

from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    cudnn_sdp_enabled,
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
)


__all__: List[str] = ["SDPBackend", "sdpa_kernel", "WARN_FOR_UNFUSED_KERNELS"]

# Note: [SDPA warnings]
# TODO: Consider using this for sdpa regardless of subclasses
# This only effects users of bias subclasses
# If this is set to True, we will warn the user if they are not using the fused kernels
# As well, it will raise warnings for all the reasons why the fused kernels can't be run.
# To set this to True, run
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
WARN_FOR_UNFUSED_KERNELS = False


# Hacks for Sphinx documentation:
# https://stackoverflow.com/questions/38765577/overriding-sphinx-autodoc-alias-of-for-import-of-private-class
SDPBackend = SDPBackend
r"""An enum-like class that contains the different backends for scaled dot product attention.
    This backend class is designed to be used with the sdpa_kernel context manager.

    The following Enums are available:
        - ERROR: An error occurred when trying to determine the backend.
        - MATH: The math backend for scaled dot product attention.
        - FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
        - EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.
        - CUDNN_ATTENTION: The cuDNN backend for scaled dot product attention.

    See :func:`torch.nn.attention.sdpa_kernel` for more details.

    .. warning:: This class is in beta and subject to change.
"""
SDPBackend.__module__ = __name__
SDPBackend.__name__ = "SDPBackend"


def _raise_kernel_warnings(params: SDPAParams) -> None:
    """
    If WARN_FOR_UNFUSED_KERNELS is set to True, this will raise warnings
    for all the reasons why the fused kernels can't be run. If using subclasses
    """
    if WARN_FOR_UNFUSED_KERNELS:
        if not can_use_efficient_attention(params):
            warn("Efficient attention can't be used because:")
            can_use_efficient_attention(params, True)
        if not can_use_flash_attention(params):
            warn("Flash attention can't be used because:")
            can_use_flash_attention(params, True)


@contextlib.contextmanager
def sdpa_kernel(backends: Union[List[SDPBackend], SDPBackend]):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backend (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.

    Example:

    .. code-block:: python

        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Only enable flash attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            scaled_dot_product_attention(...)

        # Enable the Math or Efficient attention backends
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            scaled_dot_product_attention(...)

    This context manager can be used to select which backend to use for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored, enabling all backends.
    """
    assert isinstance(
        backends, (list, SDPBackend)
    ), "Backend must be an instance of SDPBackend or a list of SDPBackend instances"

    if isinstance(backends, SDPBackend):
        backends = [backends]

    backends = set(backends)
    previous_cudnn: bool = cudnn_sdp_enabled()
    previous_flash: bool = flash_sdp_enabled()
    previous_mem_efficient: bool = mem_efficient_sdp_enabled()
    previous_math: bool = math_sdp_enabled()
    try:
        enable_cudnn = SDPBackend.CUDNN_ATTENTION in backends
        enable_flash = SDPBackend.FLASH_ATTENTION in backends
        enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION in backends
        enable_math = SDPBackend.MATH in backends

        enable_cudnn_sdp(enable_cudnn)
        enable_flash_sdp(enable_flash)
        enable_mem_efficient_sdp(enable_mem_efficient)
        enable_math_sdp(enable_math)
        yield {}
    finally:
        enable_cudnn_sdp(previous_cudnn)
        enable_flash_sdp(previous_flash)
        enable_mem_efficient_sdp(previous_mem_efficient)
        enable_math_sdp(previous_math)


def _get_flash_version() -> str:
    """This returns the closest matching tag for the flash attention backend"""
    return "2.5.6"
