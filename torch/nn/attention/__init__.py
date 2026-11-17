# mypy: allow-untyped-defs
"""This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention"""

import contextlib
from collections.abc import Iterable
from contextvars import ContextVar
from typing import Union
from warnings import warn

import torch.backends.cuda
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)


__all__: list[str] = [
    "SDPBackend",
    "sdpa_kernel",
    "WARN_FOR_UNFUSED_KERNELS",
    "register_flash_attention_impl",
    "activate_flash_attention_impl",
    "list_flash_attention_impls",
    "current_flash_attention_impl",
    "restore_flash_attention_impl",
    "enable_cudnn_sdp_python",
    "cudnn_sdp_python_enabled",
    "is_cudnn_sdp_python_available",
]


# Note: [SDPA warnings]
# TODO: Consider using this for sdpa regardless of subclasses
# This only affects users of bias subclasses
# If this is set to True, we will warn the user if they are not using the fused kernels
# As well, it will raise warnings for all the reasons why the fused kernels can't be run.
# To set this to True, run
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
WARN_FOR_UNFUSED_KERNELS = False


# SDPBackend is defined and documented in torch/csrc/Module.cpp.
SDPBackend.__module__ = __name__
SDPBackend.__name__ = "SDPBackend"
SDPBackend.__qualname__ = "SDPBackend"


def _raise_kernel_warnings(params: SDPAParams) -> None:
    """
    If WARN_FOR_UNFUSED_KERNELS is set to True, this will raise warnings
    for all the reasons why the fused kernels can't be run. If using subclasses
    """
    if WARN_FOR_UNFUSED_KERNELS:
        if not can_use_efficient_attention(params):
            warn("Efficient attention can't be used because:", stacklevel=2)
            can_use_efficient_attention(params, True)
        if not can_use_flash_attention(params):
            warn("Flash attention can't be used because:", stacklevel=2)
            can_use_flash_attention(params, True)


_backend_names = {
    "cudnn": "CUDNN_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "overrideable": "OVERRIDEABLE",
}
_sdpa_kernel_uses_priority = ContextVar("sdpa_kernel_uses_priority", default=False)


def _is_sdp_priority_order_active() -> bool:
    """Return whether an enclosing sdpa_kernel context set backend priority."""
    return _sdpa_kernel_uses_priority.get()


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


def _sdpa_kernel(backends: Iterable, set_priority: bool = False) -> None:
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
def sdpa_kernel(backends: list[SDPBackend] | SDPBackend, set_priority: bool = False):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.
        set_priority (bool=False): Whether the ordering of the backends is interpreted as their priority order.

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

        # Enable the cuDNN or flash attention backends, and in that order
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], set_priority=True
        ):
            scaled_dot_product_attention(...)

    This context manager can be used to select which backend to use for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored, enabling all backends.
    """
    if not isinstance(backends, (list, SDPBackend)):
        raise AssertionError(
            f"Backend must be an instance of SDPBackend or a list of SDPBackend instances, got {type(backends).__name__}"
        )

    if isinstance(backends, SDPBackend):
        backends = [backends]

    backends = list(dict.fromkeys(backends))

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    priority_token = _sdpa_kernel_uses_priority.set(
        set_priority or _sdpa_kernel_uses_priority.get()
    )
    try:
        _sdpa_kernel(backends, set_priority)
        yield {}
    finally:
        _sdpa_kernel_uses_priority.reset(priority_token)
        _sdpa_kernel(previous_backends, set_priority)


# variadic version of sdpa_kernel for dynamo to use while reconstructing
@contextlib.contextmanager
def _sdpa_kernel_variadic(*backends: SDPBackend):
    with sdpa_kernel(list(backends)):
        yield


def _get_flash_version() -> str:
    """This returns the closest matching tag for the flash attention backend"""
    return "2.5.7"


from . import _registry


# Re-export registry types and functions for public API
_FlashAttentionImpl = _registry._FlashAttentionImpl
_RegisterFn = _registry._RegisterFn
register_flash_attention_impl = _registry.register_flash_attention_impl
activate_flash_attention_impl = _registry.activate_flash_attention_impl
list_flash_attention_impls = _registry.list_flash_attention_impls
current_flash_attention_impl = _registry.current_flash_attention_impl
restore_flash_attention_impl = _registry.restore_flash_attention_impl

register_flash_attention_impl.__module__ = __name__
activate_flash_attention_impl.__module__ = __name__
list_flash_attention_impls.__module__ = __name__
current_flash_attention_impl.__module__ = __name__
restore_flash_attention_impl.__module__ = __name__

# Import built-in implementations to trigger self-registration
from . import _cudnn, _fa3, _fa4


def is_cudnn_sdp_python_available() -> bool:
    r"""
    .. warning:: This API is beta and subject to change.

    Returns whether ``nvidia-cudnn-frontend``'s ``cudnn.torch`` module can be
    found. It imports the ``cudnn`` package to look, but not ``cudnn.torch``,
    and does not check the version: :func:`enable_cudnn_sdp_python` still
    raises if the installed package is older than it requires.
    """
    return _cudnn.is_available()


def cudnn_sdp_python_enabled() -> bool:
    r"""
    .. warning:: This API is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is currently served by
    the cuDNN frontend Python API rather than the built-in C++ implementation.
    """
    return _cudnn.is_enabled()


def enable_cudnn_sdp_python(enabled: bool) -> None:
    r"""
    .. warning:: This API is beta and subject to change.

    Selects which implementation backs cuDNN scaled dot product attention: the
    cuDNN frontend Python API when True, or the built-in C++ implementation
    when False. Enabling requires ``nvidia-cudnn-frontend>=1.32.3``, which is
    imported on the first enable and not before.

    This chooses an *implementation*, so it is orthogonal to
    :func:`torch.backends.cuda.enable_cudnn_sdp`, which chooses whether the
    cuDNN *backend* is eligible for selection at all: with that off, neither
    implementation runs.

    It replaces the CUDA kernels of ``_scaled_dot_product_cudnn_attention``, its
    backward, and the :func:`~torch.nn.attention.varlen.varlen_attn` ops. The
    varlen ops are shared with flash attention, so with this on every varlen
    call reaches the frontend first; it serves the calls that selected cuDNN
    and passes the rest to PyTorch's own kernels, so the active flash attention
    implementation is unaffected. Nested-tensor inputs, and code that calls
    the CUDA kernels without going through the dispatcher (such as
    AOTInductor's C++ shims), keep the built-in implementation.
    """
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be a bool, got {type(enabled).__name__}")
    if enabled:
        _cudnn.enable()
    else:
        _cudnn.disable()
