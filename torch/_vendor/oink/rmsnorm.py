"""Compact public facade for the Blackwell RMSNorm implementation.

The heavy kernel implementation lives in `._rmsnorm_impl` so this module stays
small and easy to navigate while preserving the historical public import path:

    from . import rmsnorm
"""

from __future__ import annotations

from . import _rmsnorm_impl as _impl

__doc__ = _impl.__doc__

get_copy_atom_bw = _impl.get_copy_atom_bw
copy_tiled = _impl.copy_tiled
RMSNormSM100 = _impl.RMSNormSM100
rmsnorm_forward = _impl.rmsnorm_forward
rmsnorm_ref = _impl.rmsnorm_ref
fused_add_rmsnorm_forward = _impl.fused_add_rmsnorm_forward
fused_add_rmsnorm_forward_inplace = _impl.fused_add_rmsnorm_forward_inplace
fused_add_rmsnorm_inplace_ = _impl.fused_add_rmsnorm_inplace_
RMSNormBackwardSM100 = _impl.RMSNormBackwardSM100
rmsnorm_backward = _impl.rmsnorm_backward

__all__ = [
    "get_copy_atom_bw",
    "copy_tiled",
    "RMSNormSM100",
    "rmsnorm_forward",
    "rmsnorm_ref",
    "fused_add_rmsnorm_forward",
    "fused_add_rmsnorm_forward_inplace",
    "fused_add_rmsnorm_inplace_",
    "RMSNormBackwardSM100",
    "rmsnorm_backward",
]


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
