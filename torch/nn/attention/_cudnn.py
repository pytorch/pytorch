"""
cuDNN SDPA served by the cuDNN frontend Python API.

The kernels live in the out-of-tree ``nvidia-cudnn-frontend`` package. This
module swaps the cuDNN SDPA ops (``_scaled_dot_product_cudnn_attention`` and the
varlen ops) over to it, behind a single switch:
:func:`torch.backends.cuda.enable_cudnn_sdp_python`.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import TYPE_CHECKING

from . import _registry


if TYPE_CHECKING:
    from ._registry import FlashAttentionHandle as _ProviderHandle


_CUDNN_MODULE_PATH = "cudnn.torch"
ENV_VAR = "TORCH_CUDNN_SDPA_USE_PYTHON"

logger = logging.getLogger(__name__)

# Set while the provider's kernels are installed.
_ACTIVE_HANDLE: _ProviderHandle | None = None
# The provider's register_fn, taken from the flash registry; see
# _take_over_registration().
_PROVIDER_REGISTER_FN: _registry._RegisterFn | None = None


def is_available(module_path: str = _CUDNN_MODULE_PATH) -> bool:
    # find_spec imports the parent package, but not the provider itself.
    try:
        return importlib.util.find_spec(module_path) is not None
    except (ImportError, ValueError):
        return False


def is_enabled() -> bool:
    _take_over_registration()
    return _ACTIVE_HANDLE is not None


def enable(module_path: str = _CUDNN_MODULE_PATH) -> None:
    global _ACTIVE_HANDLE
    _take_over_registration()
    if _PROVIDER_REGISTER_FN is None:
        # Importing the provider registers it in the flash registry.
        importlib.import_module(module_path)
        _take_over_registration()
    if _ACTIVE_HANDLE is not None:
        return
    register_fn = _PROVIDER_REGISTER_FN
    if register_fn is None:
        raise RuntimeError(
            f"'{module_path}' did not register a 'CUDNN' implementation; "
            "a newer nvidia-cudnn-frontend is required"
        )
    _ACTIVE_HANDLE = register_fn()


def disable() -> None:
    global _ACTIVE_HANDLE
    _take_over_registration()
    handle, _ACTIVE_HANDLE = _ACTIVE_HANDLE, None
    if handle is not None:
        handle.remove()


def _take_over_registration() -> None:
    """Move the provider's "CUDNN" flash registry entry, and its activation, here.

    The package registers "CUDNN" as a flash attention impl on import. Left
    there, it is a second switch whose state this module cannot see:
    activating another flash impl would remove cuDNN's kernels, and disabling
    here would leave them installed. Every entry point above calls this first.
    """
    global _ACTIVE_HANDLE, _PROVIDER_REGISTER_FN
    register_fn = _registry._FLASH_ATTENTION_IMPLS.pop("CUDNN", None)
    if register_fn is not None:
        _PROVIDER_REGISTER_FN = register_fn
    active = _registry._FLASH_ATTENTION_ACTIVE
    if active is None or active[0] != "CUDNN":
        return
    # Adopt the installed kernels rather than reinstall them, so there is
    # nothing left to fail between removing one copy and installing another.
    _registry._FLASH_ATTENTION_ACTIVE = None
    if _ACTIVE_HANDLE is None:
        _ACTIVE_HANDLE = active[1]
    else:
        active[1].remove()


def _enable_from_env() -> None:
    """Honor TORCH_CUDNN_SDPA_USE_PYTHON; the last step of ``import torch``.

    Never raises: the variable is process-wide and usually set by a launcher,
    so a stale value or a missing package must not break ``import torch``.
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
