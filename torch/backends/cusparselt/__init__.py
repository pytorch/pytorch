# mypy: allow-untyped-defs
from typing import Optional

import torch


__all__ = [
    "version",
    "is_available",
    "get_max_alg_id",
]

try:
    from torch._C import _cusparselt
except ImportError:
    _cusparselt = None  # type: ignore[assignment]

__cusparselt_version: Optional[int] = None
__MAX_ALG_ID: Optional[int] = None

if _cusparselt is not None:

    def _init():
        global __cusparselt_version
        global __MAX_ALG_ID
        if __cusparselt_version is None:
            __cusparselt_version = _cusparselt.getVersionInt()

            # only way to get MAX_ALG_ID is to run a matmul
            A = torch.zeros(128, 128, dtype=torch.float16).cuda()
            A = torch._cslt_compress(A)
            B = torch.zeros(128, 128, dtype=torch.float16).cuda()
            _, _, _, __MAX_ALG_ID = _cusparselt.mm_search(A, B, None, None, None, False)  # type: ignore[attr-defined]
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


def get_max_alg_id() -> Optional[int]:
    r"""Return the maximum algorithm id supported by the current version of cuSPARSELt"""
    if not _init():
        return None
    return __MAX_ALG_ID
