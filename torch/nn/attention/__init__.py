# mypy: allow-untyped-defs
"""This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention"""

import contextlib
from collections.abc import Iterable
from typing import Callable, Literal, Union
from warnings import warn

import torch.backends.cuda
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)

from . import _fa4


__all__: list[str] = [
    "SDPBackend",
    "sdpa_kernel",
    "WARN_FOR_UNFUSED_KERNELS",
    "register_flash_attention_backend",
    "install_flash_attention_impl",
    "list_flash_attention_backends",
    "current_flash_attention_backend",
]


# Note: [SDPA warnings]
# TODO: Consider using this for sdpa regardless of subclasses
# This only effects users of bias subclasses
# If this is set to True, we will warn the user if they are not using the fused kernels
# As well, it will raise warnings for all the reasons why the fused kernels can't be run.
# To set this to True, run
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
WARN_FOR_UNFUSED_KERNELS = False


r"""An enum-like class that contains the different backends for scaled dot product attention.
    This backend class is designed to be used with the sdpa_kernel context manager.

    The following Enums are available:
        - ERROR: An error occurred when trying to determine the backend.
        - MATH: The math backend for scaled dot product attention.
        - FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
        - EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.
        - CUDNN_ATTENTION: The cuDNN backend for scaled dot product attention.
        - OVERRIDEABLE: The overridable backend for extension.

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


_FlashAttentionBackend = Literal["FA4"]
_FLASH_BACKEND_FA4: _FlashAttentionBackend = "FA4"

_RegisterFn = Callable[..., None]
_FLASH_ATTENTION_BACKENDS: dict[str, _RegisterFn] = {}
_FLASH_ATTENTION_ACTIVE: str | None = None


def register_flash_attention_backend(
    backend: str | _FlashAttentionBackend,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that installs a flash attention backend.

    Args:
        backend: Backend identifier (e.g., ``"FA4"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`install_flash_attention_impl`
            and should register custom kernels with the PyTorch dispatcher.
            The callable may accept optional keyword arguments such as
            ``module_path`` for configuring the backend implementation.

    Example:
        >>> def my_backend_register(module_path: str = "my_flash_impl"):
        ...     # Register custom kernels with torch dispatcher
        ...     pass  # doctest: +SKIP
        >>> register_flash_attention_backend(
        ...     "MyBackend", register_fn=my_backend_register
        ... )  # doctest: +SKIP
    """
    _FLASH_ATTENTION_BACKENDS[backend] = register_fn


def install_flash_attention_impl(
    backend: str | _FlashAttentionBackend,
) -> None:
    """
    Install into the dispatcher a previously registered flash attention backend.

    Args:
        backend: Backend identifier to activate. See
            :func:`~torch.nn.attention.list_flash_attention_backends` for available
            backends.

    Example:
        >>> install_flash_attention_impl("FA4")  # doctest: +SKIP
    """
    register_fn = _FLASH_ATTENTION_BACKENDS.get(backend)
    if register_fn is None:
        raise ValueError(f"Unknown flash attention backend '{backend}'")
    register_fn()
    global _FLASH_ATTENTION_ACTIVE
    _FLASH_ATTENTION_ACTIVE = backend


def list_flash_attention_backends() -> list[str]:
    """Return the names of all registered flash attention backends."""
    return sorted(_FLASH_ATTENTION_BACKENDS.keys())


def current_flash_attention_backend() -> str | None:
    """
    Return the currently installed flash attention backend name, if any.

    ``None`` indicates that no custom backend has been installed.
    """
    return _FLASH_ATTENTION_ACTIVE


# We are registering FA4 as a possible hot swap, but it is not actually installed in the dispatcher
# until a user calls install_flash_attention_impl("FA4")
register_flash_attention_backend(
    _FLASH_BACKEND_FA4,
    register_fn=_fa4.register_flash_attention_fa4,
)
