"""CUDA aten reduction overrides backed by the CuteDSL kernels.

Conditions check capability and fall through to ATen for unsupported calls.
Supported layouts use a fast path or the general TensorIterator decode. No out=
overload or full prod is registered, so those stay on ATen, whose results can differ
bitwise from the served overloads'.

Of the launch policies, these one-field operations reach only the one-field rules;
index and Welford rules serve direct kernel calls.

Opt out with torch.backends.python_native.cutedsl.disabled(), or per overload with
disable_operations("sum.dim_IntList", "mean.dim", "prod.dim_int", "amax", "amin");
bare "sum", "mean" or "prod" match no registration.

PYTORCH_NATIVE_INNER_TREE also orders served calls: a user who asks for the inner-tree
order gets it, and serving an unordered request in that order is fine. The condition
ignores the order, so with the variable set to "strict" a served call whose layout
cannot take the inner tree raises (with "1" it warns and runs unordered). Calls the
condition declines go to ATen, which does not follow the order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, TypeAlias

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.device import cuda_device_guard
from ...utils.lazy import LazyModule
from ...utils.tensor import const_data_ptr


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
        # Conservative: the kernels narrow their loads for lower alignment, but served
        # calls are measured and tested from 16-byte aligned bases only.
        and const_data_ptr(self) % 16 == 0
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

        trait = lambda: trait_type()(acc=cutlass.Float32)  # noqa: E731
        out_dtype = self.dtype if dtype is None else dtype
        red = _normalize_dims(dim, self.dim())
        dims = None if red is None else sorted(red)
        out = kg.replay_reduce(trait, key, self, dims, out_dtype)
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
