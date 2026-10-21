"""
cuDNN SDPA served by the cuDNN frontend Python API.

The kernels live in the out-of-tree ``nvidia-cudnn-frontend`` package. This
module swaps the cuDNN SDPA ops (``_scaled_dot_product_cudnn_attention`` and the
varlen ops) over to it, behind a single switch:
:func:`torch.backends.cuda.enable_cudnn_sdp_python`.

The package's ``install_sdpa_overrides()`` installs the overrides and returns a
handle whose ``remove()`` takes them out again. This module holds that handle,
so whether it is set is the one record of whether the switch is on.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import Protocol


_CUDNN_MODULE_PATH = "cudnn.torch"
_INSTALL_FN = "install_sdpa_overrides"
# The first nvidia-cudnn-frontend release with install_sdpa_overrides().
_MIN_FRONTEND_VERSION = "1.32"
ENV_VAR = "TORCH_CUDNN_SDPA_USE_PYTHON"

logger = logging.getLogger(__name__)


class _OverridesHandle(Protocol):
    def remove(self) -> None: ...


# Set while the provider's kernels are installed.
_ACTIVE_HANDLE: _OverridesHandle | None = None


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
    provider = importlib.import_module(module_path)
    install = getattr(provider, _INSTALL_FN, None)
    if install is None:
        raise RuntimeError(
            f"'{module_path}' has no {_INSTALL_FN}(); "
            f"nvidia-cudnn-frontend>={_MIN_FRONTEND_VERSION} is required"
        )
    handle = install()
    if not callable(getattr(handle, "remove", None)):
        raise TypeError(
            f"{module_path}.{_INSTALL_FN}() returned {handle!r}, "
            "which has no remove() to undo it"
        )
    _ACTIVE_HANDLE = handle


def disable() -> None:
    global _ACTIVE_HANDLE
    handle, _ACTIVE_HANDLE = _ACTIVE_HANDLE, None
    if handle is not None:
        handle.remove()


def _enable_from_env() -> None:
    """Honor TORCH_CUDNN_SDPA_USE_PYTHON; the last step of ``import torch``.

    Never raises: the variable is process-wide and usually set by a launcher,
    so a stale value or a missing package must not break ``import torch``.
    """
    if os.environ.get(ENV_VAR, "0") != "1":
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
