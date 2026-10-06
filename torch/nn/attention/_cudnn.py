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
# The provider's register_fn, captured once; see _capture_provider().
_PROVIDER_REGISTER_FN: _registry._RegisterFn | None = None


def is_available(module_path: str = _CUDNN_MODULE_PATH) -> bool:
    # find_spec imports the parent package, but not the provider itself.
    try:
        return importlib.util.find_spec(module_path) is not None
    except (ImportError, ValueError):
        return False


def is_enabled() -> bool:
    return _ACTIVE_HANDLE is not None


def enable(module_path: str = _CUDNN_MODULE_PATH) -> None:
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is not None:
        return
    register_fn = _PROVIDER_REGISTER_FN
    if register_fn is None:
        register_fn = _capture_provider(module_path)
    _ACTIVE_HANDLE = register_fn()


def disable() -> None:
    global _ACTIVE_HANDLE
    handle, _ACTIVE_HANDLE = _ACTIVE_HANDLE, None
    if handle is not None:
        handle.remove()


def _capture_provider(module_path: str) -> _registry._RegisterFn:
    """Import the provider and take its register_fn out of the flash registry.

    The package registers "CUDNN" as a flash attention impl on import. Left
    there, it would be a second switch whose state this module cannot see:
    activating another flash impl would remove cuDNN's kernels, and disabling
    here would leave the registry reporting "CUDNN" as active.
    """
    global _PROVIDER_REGISTER_FN
    importlib.import_module(module_path)
    register_fn = _registry._FLASH_ATTENTION_IMPLS.pop("CUDNN", None)
    if register_fn is None:
        raise RuntimeError(
            f"'{module_path}' did not register a 'CUDNN' implementation; "
            "a newer nvidia-cudnn-frontend is required"
        )
    # Already activated through the registry: uninstall it so only one copy of
    # the kernels is ever installed, owned here.
    active = _registry._FLASH_ATTENTION_ACTIVE
    if active is not None and active[0] == "CUDNN":
        _registry.restore_flash_attention_impl()
    _PROVIDER_REGISTER_FN = register_fn
    return register_fn


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
