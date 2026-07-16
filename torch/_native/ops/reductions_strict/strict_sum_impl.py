"""CuTeDSL strict-numerics `sum` override for the aten sum operator.

Gated on ``torch._inductor.config.numerics == "strict"`` -- the SAME flag that drives
torch.compile/Inductor -- so eager ``torch.sum`` and ``torch.compile`` are bitwise-identical.
Follows ops/norm's RMSNorm pattern: a ``cond`` (per-call routing) + ``impl`` (the kernel),
registered via ``cutedsl_utils.register_op_override`` which handles runtime/version gating
and the recursion-free aten fallthrough. CuTeDSL is imported lazily inside the kernel.
"""

from __future__ import annotations

import torch

from ... import cutedsl_utils as cu
from .kernels import SUPPORTED_DTYPES, strict_sum


# cached torch._inductor.config module (the numerics flag lives there); avoid re-import per call
_IC = None


def _strict_enabled() -> bool:
    global _IC
    if _IC is None:
        try:
            import torch._inductor.config as ic

            _IC = ic
        except Exception:
            return False
    return getattr(_IC, "numerics", "default") == "strict"


def _is_supported(x: torch.Tensor) -> bool:
    if x.device.type != "cuda":
        return False
    if torch.version.hip is not None:
        return False
    return x.dtype in SUPPORTED_DTYPES


def _dims_ok(dim, nd: int) -> bool:
    # Invalid/duplicate dims -> fall through to aten so it raises the correct
    # IndexError / duplicate-dim RuntimeError rather than silently reducing the wrong axis.
    if dim is None:
        return True
    dims = dim if isinstance(dim, (tuple, list)) else [dim]
    seen: set = set()
    for d in dims:
        if not isinstance(d, int) or d < -nd or d >= nd:
            return False
        n = d % nd if nd else 0
        if n in seen:
            return False
        seen.add(n)
    return True


def _sum_cond(self, dim=None, keepdim=False, *, dtype=None) -> bool:
    # Only route when strict is on and the input is eligible; otherwise fall through to aten.
    # `dtype=` requests an output cast aten handles -> defer. cond uses only torch dtypes
    # (no cutlass), so it stays cheap and import-safe.
    if dtype is not None:
        return False
    if not _strict_enabled():
        return False
    if not _is_supported(self):
        return False
    if self.numel() == 0:
        return False
    if not _dims_ok(dim, self.dim()):
        return False
    return True


def _sum_impl(self, dim=None, keepdim=False, *, dtype=None) -> torch.Tensor:
    return strict_sum(self, dim, keepdim)


def register_reduction_overrides() -> None:
    # Safe to call unconditionally at import: register_op_override short-circuits when the
    # CuTeDSL runtime/version is unavailable (and never calls cuInit / poisons fork).
    cu.register_op_override(
        "aten",
        "sum.dim_IntList",
        "CUDA",
        cond=_sum_cond,
        impl=_sum_impl,
    )


def is_available() -> bool:
    """True when the CuTeDSL runtime is usable (i.e. the override is active). For test skips."""
    return cu.runtime_available()
