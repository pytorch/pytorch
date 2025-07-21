# mypy: allow-untyped-defs
"""This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention"""

import contextlib
from collections.abc import Iterable
from typing import Union
from warnings import warn

import torch.backends.cuda
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)


__all__: list[str] = ["SDPBackend", "sdpa_kernel", "WARN_FOR_UNFUSED_KERNELS"]

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


_backend_names = {
    "cudnn": "CUDNN_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "overrideable": "OVERRIDEABLE",
}


def _backend_from_string(name: str):
    return getattr(SDPBackend, name)


def _cur_sdpa_kernel_backends(with_priority: bool = False):
    backends = []
    for name, val in _backend_names.items():
        if getattr(torch._C, f"_get_{name}_sdp_enabled")():
            backends.append(getattr(SDPBackend, val))
    if with_priority:
        curr_priority = torch._C._get_sdp_priority_order()
        backends = sorted(
            backends, key=lambda backend: curr_priority.index(int(backend))
        )
    return backends


def _sdpa_kernel(backends: Iterable, set_priority: bool = False):
    for name, val in _backend_names.items():
        enabled = getattr(SDPBackend, val) in backends
        getattr(torch._C, f"_set_sdp_use_{name}")(enabled)
    if set_priority:
        # backends should be a unique list
        user_priority = [int(backend) for backend in backends]
        previous_priority = torch._C._get_sdp_priority_order()
        for backend in previous_priority:
            if backend not in user_priority:
                user_priority.append(int(backend))
        torch._C._set_sdp_priority_order(user_priority)


@contextlib.contextmanager
def sdpa_kernel(
    backends: Union[list[SDPBackend], SDPBackend], set_priority: bool = False
):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.
        set_priority_order (bool=False): Whether the ordering of the backends is interpreted as their priority order.

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
    assert isinstance(backends, (list, SDPBackend)), (
        "Backend must be an instance of SDPBackend or a list of SDPBackend instances"
    )

    if isinstance(backends, SDPBackend):
        backends = [backends]

    backends = list(dict.fromkeys(backends))

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    try:
        _sdpa_kernel(backends, set_priority)
        yield {}
    finally:
        _sdpa_kernel(previous_backends, set_priority)


# variadic version of sdpa_kernel for dynamo to use while reconstructing
@contextlib.contextmanager
def _sdpa_kernel_variadic(*backends: SDPBackend):
    with sdpa_kernel(list(backends)):
        yield


def _get_flash_version() -> str:
    """This returns the closest matching tag for the flash attention backend"""
    return "2.5.7"
