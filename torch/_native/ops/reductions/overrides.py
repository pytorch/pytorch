"""CUDA aten reduction overrides backed by the CuteDSL kernels.

Conditions check capability and fall through to ATen for unsupported calls.
Supported layouts use a fast path or the general TensorIterator decode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, TypeAlias

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.device import cuda_device_guard
from ...utils.lazy import LazyModule


if TYPE_CHECKING:
    from . import kernel_general as kg, traits as T
else:
    # T and kg import `cutlass`, which `import torch` must not do (see
    # test_no_dsl_imports_after_import_torch). Only the *_impl functions touch them.
    T = LazyModule("torch._native.ops.reductions.traits")
    kg = LazyModule("torch._native.ops.reductions.kernel_general")


_Dim: TypeAlias = int | Sequence[int] | None
_Red: TypeAlias = set[int] | None


# Compute-capability majors this family's kernels have been run on: Hopper and Blackwell.
_ARCH_MAJORS = (9, 10)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _normalize_dims(dim: _Dim, ndim: int) -> _Red:
    """Normalize aten dimensions, returning None for reduce-all."""
    if dim is None:
        return None
    dims = [dim] if isinstance(dim, int) else list(dim)
    if not dims:
        return None
    return {d % ndim for d in dims}


def _dims_ok(dim: _Dim, ndim: int) -> bool:
    if ndim == 0 or ndim > 64:
        return False
    if dim is None:
        return True
    dims = [dim] if isinstance(dim, int) else list(dim)
    if not dims:
        return True
    if not all(-ndim <= d < ndim for d in dims):
        return False
    norm = [d % ndim for d in dims]
    return len(set(norm)) == len(norm)


def _keepdim_reshape(
    out: torch.Tensor,
    x_shape: Sequence[int],
    red: _Red,
    keepdim: bool,
) -> torch.Tensor:
    if not keepdim:
        return out
    target = (
        [1] * len(x_shape)
        if red is None
        else [1 if i in red else s for i, s in enumerate(x_shape)]
    )
    return out.reshape(target)


def _out_ok(self: torch.Tensor, dtype: torch.dtype | None) -> bool:
    # The kernels accumulate the input in fp32. ATen casts the input to any other
    # requested dtype first, so only these requests keep ATen's results.
    narrow = (torch.float16, torch.bfloat16)
    return (
        dtype is None
        or dtype == self.dtype
        or (dtype == torch.float32 and self.dtype in narrow)
    )


def _base_cond(self: torch.Tensor, dim: _Dim) -> bool:
    # A condition must decline unsupported calls without raising.
    return (
        not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and self.dtype in _SUPPORTED_DTYPES
        and not self.is_neg()
        and not self.is_conj()
        and self.const_data_ptr() % 16 == 0  # type: ignore[attr-defined]
        and _dims_ok(dim, self.dim())
        and self.numel() != 0
    )


def _cond(self: torch.Tensor, dim: _Dim = None, keepdim: bool = False) -> bool:
    return _base_cond(self, dim)


def _dtype_cond(
    self: torch.Tensor,
    dim: _Dim = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> bool:
    return _base_cond(self, dim) and _out_ok(self, dtype)


def _make_impl(trait_type: Callable[[], Any], key: str) -> Callable[..., torch.Tensor]:
    def impl(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # Lazy: importing cutlass at module import would pull the DSL into `import torch`.
        import cutlass

        trait = trait_type()(acc=cutlass.Float32)
        out_dtype = self.dtype if dtype is None else dtype
        red = _normalize_dims(dim, self.dim())
        if red is None:
            out = kg.reduce_all(trait, key, self, out_dtype)
        else:
            out = kg.reduce_dim(trait, key, self, sorted(red), out_dtype)
        return _keepdim_reshape(out, self.shape, red, keepdim)

    return impl


_sum_impl = _make_impl(lambda: T.SumOps, "sum")
_mean_impl = _make_impl(lambda: T.MeanOps, "mean")
_prod_impl = _make_impl(lambda: T.ProdOps, "prod")
_amax_impl = _make_impl(lambda: T.AMaxOps, "amax")
_amin_impl = _make_impl(lambda: T.AMinOps, "amin")


def register_reduction_overrides() -> None:
    # cu.register_op_override is a no-op when the CuteDSL runtime is unavailable.
    overrides = (
        ("sum.dim_IntList", _dtype_cond, _sum_impl),
        ("mean.dim", _dtype_cond, _mean_impl),
        ("amax", _cond, _amax_impl),
        ("amin", _cond, _amin_impl),
        ("prod.dim_int", _dtype_cond, _prod_impl),
    )
    for op, cond, impl in overrides:
        impl = cuda_device_guard(impl)
        cu.register_op_override("aten", op, "CUDA", cond=cond, impl=impl)
