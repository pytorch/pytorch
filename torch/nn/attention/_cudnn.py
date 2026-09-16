"""
cuDNN attention implementation, via the cuDNN frontend Python API.

The kernels live in the out-of-tree ``nvidia-cudnn-frontend`` package, which
registers itself into the flash attention registry when imported. This module
owns the lifecycle for swapping cuDNN's aten kernels over to that package, and
exposes it two ways:

* :func:`torch.backends.cuda.enable_cudnn_sdp_python` -- the cuDNN backend's
  own switch, independent of which flash implementation is active;
* the ``"CUDNN"`` entry in the flash attention registry, for parity with
  FA3/FA4 and :func:`torch.nn.attention.activate_flash_attention_impl`.

Both routes share one handle, so enabling twice installs once and either route
can turn it off.

Note that this replaces the implementation of the *cuDNN* SDPA ops
(``_scaled_dot_product_cudnn_attention`` and the varlen ops), not the flash
ops. It is orthogonal to :func:`torch.backends.cuda.enable_cudnn_sdp`, which
controls whether the cuDNN backend is eligible for selection at all.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import importlib.util
import logging
import os

from . import _registry
from ._registry import FlashAttentionHandle as _ProviderHandle


__all__ = [
    "register_cudnn_attention",
]


_CUDNN_MODULE_PATH = "cudnn.torch"

ENV_VAR = "TORCH_CUDNN_SDPA_USE_PYTHON"

logger = logging.getLogger(__name__)

# The provider's handle while its kernels are installed; None when they are not.
_ACTIVE_HANDLE: _ProviderHandle | None = None

# The provider's own register_fn, captured on first import. Held here so both
# entry points install through this module rather than one of them calling the
# provider directly -- see _capture_provider().
_PROVIDER_REGISTER_FN = None


def is_available(module_path: str = _CUDNN_MODULE_PATH) -> bool:
    """Whether the provider package can be imported.

    Locating a submodule imports its parent package, so this is cheap but not
    entirely free; it does not import the provider itself.
    """
    try:
        return importlib.util.find_spec(module_path) is not None
    except (ImportError, ValueError):
        return False


def is_enabled() -> bool:
    """Whether cuDNN SDPA is currently served by the Python implementation."""
    return _ACTIVE_HANDLE is not None


def enable(module_path: str = _CUDNN_MODULE_PATH) -> None:
    """Install the provider's kernels over the cuDNN SDPA ops.

    Idempotent: a second call is a no-op rather than a second install, so the
    registry route and the ``torch.backends.cuda`` route cannot get out of step
    with each other. Use :func:`disable` to undo -- the handle is held here
    rather than handed out, so there is one owner of the installed state.
    """
    global _ACTIVE_HANDLE

    if _ACTIVE_HANDLE is not None:
        return

    if _PROVIDER_REGISTER_FN is None:
        _capture_provider(module_path)

    _ACTIVE_HANDLE = _PROVIDER_REGISTER_FN()


def _capture_provider(module_path: str) -> None:
    """Import the provider and take over the registry entry it installs.

    The package registers "CUDNN" with its own callable on import. Leaving that
    in place would let ``activate_flash_attention_impl("CUDNN")`` call the
    provider directly, bypassing this module -- the registry would then hold a
    handle this module does not know about, and the two views of "is the Python
    implementation active" would drift apart. Capturing the callable and
    restoring our own entry keeps both routes funnelling through :func:`enable`.
    """
    global _PROVIDER_REGISTER_FN

    importlib.import_module(module_path)

    register_fn = _registry._FLASH_ATTENTION_IMPLS.get("CUDNN")
    if register_fn is None or register_fn is register_cudnn_attention:
        # The package imported but did not take over the registration, so it
        # predates the flash-impl registry. Say so, rather than recursing.
        raise RuntimeError(
            f"'{module_path}' did not register a 'CUDNN' flash attention "
            f"implementation; a newer nvidia-cudnn-frontend is required"
        )

    _PROVIDER_REGISTER_FN = register_fn
    _registry.register_flash_attention_impl(
        "CUDNN", register_fn=register_cudnn_attention
    )


def disable() -> None:
    """Restore the built-in C++ implementation of the cuDNN SDPA ops."""
    global _ACTIVE_HANDLE

    handle, _ACTIVE_HANDLE = _ACTIVE_HANDLE, None
    if handle is not None:
        handle.remove()


class _CuDNNRegistryHandle:
    """Registry-facing handle; restoring through the registry also clears the
    ``torch.backends.cuda`` view, so the two routes agree."""

    def remove(self) -> None:
        disable()


def register_cudnn_attention(module_path: str = _CUDNN_MODULE_PATH):
    """
    Register cuDNN attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the cuDNN frontend torch provider.

    Unlike FA3/FA4, the implementation is not shimmed here: importing
    ``module_path`` re-registers "CUDNN" with the provider's own callable, and
    :func:`enable` hands off to it. That keeps the kernel set (dense SDPA plus
    the varlen ops) owned entirely by the package that ships the kernels, so a
    provider update does not need a matching change here.
    """
    enable(module_path)
    return _CuDNNRegistryHandle()


def _enable_from_env() -> None:
    """Honor ``TORCH_CUDNN_SDPA_USE_PYTHON`` at ``torch.nn.attention`` import.

    Lets an existing script run on the cuDNN Python implementation without a
    source change, which is what makes A/B comparison and soak testing
    practical.

    Never raises. The variable is process-wide and typically set by a job
    launcher, so a stale value, a missing package, or an unfriendly import
    order must not break ``import torch``; it warns and leaves the built-in C++
    implementation in place. Enabling does import the provider during
    ``import torch`` -- that cost is the point of having asked for it.
    """
    if os.environ.get(ENV_VAR, "") not in ("1", "true", "True"):
        return
    try:
        enable()
    except Exception:
        logger.warning(
            "%s is set but the cuDNN Python implementation could not be enabled; "
            "the built-in implementation stays active.",
            ENV_VAR,
            exc_info=True,
        )


_registry.register_flash_attention_impl(
    "CUDNN", register_fn=register_cudnn_attention
)
