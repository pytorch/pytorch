"""CUDA aten reduction overrides backed by the CuteDSL kernels.

Conditions check capability and fall through to ATen for unsupported calls.
Supported layouts use a fast path or the general TensorIterator decode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TYPE_CHECKING, TypeAlias

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
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
_STORABLE_DTYPES = _SUPPORTED_DTYPES


def _acc_policy() -> dict[torch.dtype, tuple[Any, torch.dtype]]:
    # Lazy: the values are `cutlass` dtypes, so resolving at import would pull cutlass in.
    import cutlass

    return {
        torch.float16: (cutlass.Float32, torch.float32),
        torch.bfloat16: (cutlass.Float32, torch.float32),
        torch.float32: (cutlass.Float32, torch.float32),
    }


def _acc_for_dtype(dtype: torch.dtype, widen: bool = True) -> tuple[Any, torch.dtype]:
    return _acc_policy()[dtype]


def _acc_for(self: torch.Tensor, widen: bool = True) -> tuple[Any, torch.dtype]:
    return _acc_for_dtype(self.dtype, widen)


def _kernel_input(self: torch.Tensor, canonical_bool: bool = True) -> torch.Tensor:
    return self


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


class _Envelope(NamedTuple):
    inputs: tuple[torch.dtype, ...]
    outs: tuple[torch.dtype, ...]


_FLOAT_ONLY = _Envelope(_SUPPORTED_DTYPES, _SUPPORTED_DTYPES)
_NO_OUT_DTYPE = _Envelope(_SUPPORTED_DTYPES, ())

_ENVELOPE = {
    "sum": _FLOAT_ONLY,
    "mean": _FLOAT_ONLY,
    "prod": _FLOAT_ONLY,
    "amax": _NO_OUT_DTYPE,
    "amin": _NO_OUT_DTYPE,
}


def _env(key: str) -> _Envelope:
    return _ENVELOPE[key]


def _out_ok(key: str, dtype: torch.dtype | None) -> bool:
    return dtype is None or dtype in _env(key).outs


def _out_dtype(self: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    return self.dtype if dtype is None else dtype


def _sum_out_dtype(self: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    return self.dtype if self.dtype.is_floating_point else torch.int64


def _base_cond(self: torch.Tensor, dim: _Dim, key: str) -> bool:
    # A condition must decline unsupported calls without raising.
    return (
        not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and self.dtype in _env(key).inputs
        and cap.on_current_device(self)
        and not self.is_neg()
        and not self.is_conj()
        and self.const_data_ptr() % 16 == 0  # type: ignore[attr-defined]
        and _dims_ok(dim, self.dim())
        and self.numel() != 0
    )


def _make_cond(key: str) -> Callable[..., bool]:
    def cond(self: torch.Tensor, dim: _Dim = None, keepdim: bool = False) -> bool:
        return _base_cond(self, dim, key)

    return cond


def _make_dtype_cond(key: str) -> Callable[..., bool]:
    def cond(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> bool:
        return _base_cond(self, dim, key) and _out_ok(key, dtype)

    return cond


def _run1(
    make_trait: Callable[[Any], Any],
    key: str,
    self: torch.Tensor,
    red: _Red,
    keepdim: bool,
    out_torch_dtype: torch.dtype,
    widen: bool = True,
    canonical_bool: bool = True,
    acc_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x = _kernel_input(self, canonical_bool)
    acc, kout = (
        _acc_for(x, widen) if acc_dtype is None else _acc_for_dtype(acc_dtype, widen)
    )
    trait = make_trait(acc)
    kern_out = out_torch_dtype if out_torch_dtype in _STORABLE_DTYPES else kout
    if red is None:
        out = kg.reduce_all(trait, key, x, kern_out)
    else:
        out = kg.reduce_dim(trait, key, x, sorted(red), kern_out)
    out = _keepdim_reshape(out, self.shape, red, keepdim)
    return out.to(out_torch_dtype)


def _make_impl(
    trait_type: Callable[[], Any],
    key: str,
    out_dtype: Callable[[torch.Tensor, torch.dtype | None], torch.dtype],
    *,
    widen: bool = True,
    canonical_bool: bool = True,
) -> Callable[..., torch.Tensor]:
    def impl(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return _run1(
            lambda acc: trait_type()(acc=acc),
            key,
            self,
            _normalize_dims(dim, self.dim()),
            keepdim,
            out_dtype(self, dtype),
            widen=widen,
            canonical_bool=canonical_bool,
        )

    return impl


_sum_impl = _make_impl(lambda: T.SumOps, "sum", _sum_out_dtype)
_mean_impl = _make_impl(lambda: T.MeanOps, "mean", _out_dtype)
_prod_impl = _make_impl(lambda: T.ProdOps, "prod", _sum_out_dtype)
_amax_impl = _make_impl(
    lambda: T.AMaxOps, "amax", _out_dtype, widen=False, canonical_bool=False
)
_amin_impl = _make_impl(
    lambda: T.AMinOps, "amin", _out_dtype, widen=False, canonical_bool=False
)


def register_reduction_overrides() -> None:
    # cu.register_op_override is a no-op when the CuteDSL runtime is unavailable.
    overrides = (
        ("sum.dim_IntList", _make_dtype_cond("sum"), _sum_impl),
        ("mean.dim", _make_dtype_cond("mean"), _mean_impl),
        ("amax", _make_cond("amax"), _amax_impl),
        ("amin", _make_cond("amin"), _amin_impl),
        ("prod.dim_int", _make_dtype_cond("prod"), _prod_impl),
    )
    for op, cond, impl in overrides:
        cu.register_op_override("aten", op, "CUDA", cond=cond, impl=impl)
