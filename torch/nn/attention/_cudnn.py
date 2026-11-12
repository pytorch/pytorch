"""
cuDNN SDPA served by the cuDNN frontend Python API.

The kernels live in the out-of-tree ``nvidia-cudnn-frontend`` package. This
module swaps the cuDNN SDPA ops (``_scaled_dot_product_cudnn_attention`` and the
varlen ops) over to it, behind a single switch:
:func:`torch.nn.attention.enable_cudnn_sdp_python`.

The package's ``install_sdpa_overrides()`` installs the overrides and returns a
handle whose ``remove()`` takes them out again. This module holds that handle,
so whether it is set is the one record of whether the switch is on.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Protocol

from torch._vendor.packaging.version import InvalidVersion, Version


_CUDNN_MODULE_PATH = "cudnn.torch"
_INSTALL_FN = "install_sdpa_overrides"
# The first nvidia-cudnn-frontend release that both honors the backend
# varlen_attn selected and looks for this switch in torch.nn.attention. Older
# releases register "CUDNN" in the flash attention registry on import, a second
# switch, so they must be rejected before cudnn.torch is imported.
_MIN_FRONTEND_VERSION = "1.32.3"


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


def _check_frontend_version(module_path: str) -> None:
    # Imports only the parent package, never the provider.
    package = importlib.import_module(module_path.partition(".")[0])
    found = getattr(package, "__version__", None)
    try:
        ok = isinstance(found, str) and Version(found) >= Version(_MIN_FRONTEND_VERSION)
    except InvalidVersion:
        ok = False
    if not ok:
        raise RuntimeError(
            f"nvidia-cudnn-frontend>={_MIN_FRONTEND_VERSION} is required; "
            f"'{module_path}' comes from version {found}"
        )


def enable(module_path: str = _CUDNN_MODULE_PATH) -> None:
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is not None:
        return
    _check_frontend_version(module_path)
    provider = importlib.import_module(module_path)
    handle = getattr(provider, _INSTALL_FN)()
    if not callable(getattr(handle, "remove", None)):
        raise TypeError(
            f"{module_path}.{_INSTALL_FN}() returned {handle!r}, "
            "which has no remove() to undo it"
        )
    _ACTIVE_HANDLE = handle


def disable() -> None:
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is not None:
        # Forget the handle only once its overrides are gone, so a failed
        # remove() leaves the switch on and retryable.
        _ACTIVE_HANDLE.remove()
        _ACTIVE_HANDLE = None
