# mypy: allow-untyped-defs
from typing import Optional

import torch


__all__ = [
    "version",
    "is_available",
]

try:
    from torch._C import _cusparselt
except ImportError:
    _cusparselt = None  # type: ignore[assignment]

__cusparselt_version: Optional[int] = None

if _cusparselt is not None:

    def _init():
        global __cusparselt_version
        if __cusparselt_version is None:
            __cusparselt_version = _cusparselt.getVersionInt()
        return True

else:

    def _init():
        return False


def version() -> Optional[int]:
    """Return the version of cuSPARSELt"""
    if not _init():
        return None
    return __cusparselt_version


def is_available() -> bool:
    r"""Return a bool indicating if cuSPARSELt is currently available."""
    return torch._C._has_cusparselt
