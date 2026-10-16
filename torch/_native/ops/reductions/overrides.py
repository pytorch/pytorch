"""CUDA aten reduction overrides backed by the CuteDSL kernels.

Conditions check capability and fall through to ATen for unsupported calls.
Supported layouts use a fast path or the general TensorIterator decode.
"""

from __future__ import annotations

import math
import warnings
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
_Result: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


# Compute-capability majors this family's kernels have been run on: Hopper and Blackwell.
_ARCH_MAJORS = (9, 10)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_STORABLE_DTYPES = _SUPPORTED_DTYPES


def _acc_policy() -> dict[torch.dtype, tuple[Any, torch.dtype]]:
    # Lazy: the values are `cutlass` dtypes, so resolving at import would pull cutlass in.
    # INPUT torch dtype -> (accumulator cute dtype, that accumulator's torch dtype).
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
    if not dims or ndim == 0:
        return None
    return {d % ndim for d in dims}


def _dims_ok(dim: _Dim, ndim: int) -> bool:
    if ndim > 64:
        return False
    if dim is None:
        return True
    dims = [dim] if isinstance(dim, int) else list(dim)
    if ndim == 0:
        return len(dims) <= 1 and all(d in (0, -1) for d in dims)
    if not all(-ndim <= d < ndim for d in dims):
        return False
    norm = [d % ndim for d in dims]
    return len(set(norm)) == len(norm)


_INT32_LIMIT = 1 << 31


# What ATen returns for an EMPTY reduced axis, per trait key. An ABSENT key has no
# identity and ATen raises: the max/min family and the +-inf norms.
_EMPTY_ID = {
    "sum": 0,
    "nansum": 0,
    "prod": 1,
    "mean": float("nan"),
    "var": float("nan"),
    "std": float("nan"),
    "var_mean": float("nan"),
    "std_mean": float("nan"),
    "cnz": 0,
    "all": True,
    "any": False,
}


def _empty_id_for(key: str) -> Any:
    if key.startswith("vnorm"):
        # vector_norm's key carries its ord. A finite p is an empty sum of |x|**p, so 0;
        # +-inf has no identity ("cannot compute the inf norm on the dimension").
        return 0 if math.isfinite(float(key[len("vnorm") :])) else None
    return _EMPTY_ID.get(key)


def _reduced_shape(x_shape: Sequence[int], red: _Red, keepdim: bool) -> list[int]:
    """The output shape of reducing `red` (None = every axis) out of `x_shape`."""
    if red is None:
        return [1] * len(x_shape) if keepdim else []
    if keepdim:
        return [1 if i in red else s for i, s in enumerate(x_shape)]
    return [s for i, s in enumerate(x_shape) if i not in red]


def _empty_ok(x: torch.Tensor, dim: _Dim, empty_id: Any) -> bool:
    # Serviceable when a KEPT extent is zero (the output is empty, so no identity is needed
    # -- true even of amax) or when the op HAS one.
    kept = _reduced_shape(x.shape, _normalize_dims(dim, x.dim()), False)
    return math.prod(kept) == 0 or empty_id is not None


def _empty_result(
    self: torch.Tensor,
    red: _Red,
    keepdim: bool,
    dtypes: Sequence[torch.dtype],
    fills: Sequence[Any],
) -> tuple[torch.Tensor, ...]:
    # Nothing to launch: an allocation (empty output) or a fill (the identity over the kept
    # shape). torch.full covers both -- over an empty shape it writes nothing.
    shape = _reduced_shape(self.shape, red, keepdim)
    return tuple(
        torch.full(shape, 0 if f is None else f, dtype=d, device=self.device)
        for d, f in zip(dtypes, fills)
    )


def _keepdim_reshape(
    out: torch.Tensor,
    x_shape: Sequence[int],
    red: _Red,
    keepdim: bool,
) -> torch.Tensor:
    if keepdim:
        return kg._as_shape(out, _reduced_shape(x_shape, red, True))
    return out


# ----------------------------------------------------------- shared cond helpers


def _out_dtype(self: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    # aten's rule for a value reduction: the explicit `dtype` if given, else the input's own.
    return dtype if dtype is not None else self.dtype


class _Envelope(NamedTuple):
    inputs: tuple[torch.dtype, ...]
    outs: tuple[torch.dtype, ...]


# THE DTYPE ENVELOPE, one row per op: widening a dtype is a row edit, not a pass over
# every cond. Empty `outs` means no `dtype=` argument; vector_norm's key carries its ord.
_FLOAT_ONLY = _Envelope(_SUPPORTED_DTYPES, _SUPPORTED_DTYPES)
_NO_OUT_DTYPE = _Envelope(_SUPPORTED_DTYPES, ())

_ENVELOPE = {
    "sum": _FLOAT_ONLY,
    "nansum": _FLOAT_ONLY,
    "prod": _FLOAT_ONLY,
    "mean": _FLOAT_ONLY,
    "vnorm": _FLOAT_ONLY,
    "var": _NO_OUT_DTYPE,
    "std": _NO_OUT_DTYPE,
    "var_mean": _NO_OUT_DTYPE,
    "std_mean": _NO_OUT_DTYPE,
    "amax": _NO_OUT_DTYPE,
    "amin": _NO_OUT_DTYPE,
    "max_dim": _NO_OUT_DTYPE,
    "min_dim": _NO_OUT_DTYPE,
    "aminmax": _NO_OUT_DTYPE,
    "argmax": _NO_OUT_DTYPE,
    "argmin": _NO_OUT_DTYPE,
    "all": _NO_OUT_DTYPE,
    "any": _NO_OUT_DTYPE,
    "cnz": _NO_OUT_DTYPE,
}


def _env(key: str) -> _Envelope:
    # vector_norm's key is f"vnorm{ord}"; every other key is the op name.
    return _ENVELOPE[key[: len("vnorm")] if key.startswith("vnorm") else key]


def _out_ok(key: str, dtype: torch.dtype | None) -> bool:
    # An explicit `dtype=` is deliverable when the row lists it: we accumulate wide and narrow at
    # the store, so the check is membership rather than a cast rule.
    return dtype is None or dtype in _env(key).outs


def _sum_out_dtype(self: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    # aten's rule for the SUM-like ops: an explicit `dtype` wins, a float keeps its own, and
    # an INTEGRAL input promotes to int64 (torch.sum on int32 returns int64).
    if dtype is not None:
        return dtype
    return self.dtype if self.dtype.is_floating_point else torch.int64


def _base_cond(self: torch.Tensor, dim: _Dim, key: str) -> bool:
    # Shared capability gate; never raises, since a throwing cond crashes the dispatcher
    # instead of falling back. Complex inputs are outside the dtype envelope, so only CONJ
    # needs an explicit metadata-bit check. Implementations resolve lazy negative inputs.
    return (
        not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and self.dtype in _env(key).inputs
        and not self.is_conj()
        and _dims_ok(dim, self.dim())
        and (self.numel() != 0 or _empty_ok(self, dim, _empty_id_for(key)))
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


def _make_dof_cond(key: str) -> Callable[..., bool]:
    def cond(
        self: torch.Tensor,
        dim: _Dim = None,
        *,
        correction: Any = None,
        keepdim: bool = False,
    ) -> bool:
        return _base_cond(self, dim, key) and _dof_supported(correction)

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
    # Every single-output reduction override funnels through here: the accumulator comes from
    # _acc_policy, and the result is aten's output dtype for this op.
    if self.numel() == 0:
        fill = _empty_id_for(key)
        return _empty_result(self, red, keepdim, [out_torch_dtype], [fill])[0]
    x = _kernel_input(self, canonical_bool)
    acc, kout = (
        _acc_for(x, widen) if acc_dtype is None else _acc_for_dtype(acc_dtype, widen)
    )
    trait = make_trait(acc)
    # STORE aten's output dtype where the store can produce it: casting the fp32 accumulator
    # afterwards is a whole extra ~2us launch per call. Accumulation is untouched.
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
    cast_input: bool = True,
) -> Callable[..., torch.Tensor]:
    def impl(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # NanSum keeps the source type to identify NaNs and changes its accumulator.
        x = (
            self
            if not cast_input or dtype is None or dtype is self.dtype
            else self.to(dtype)
        )
        return _run1(
            lambda acc: trait_type()(acc=acc),
            key,
            x,
            _normalize_dims(dim, x.dim()),
            keepdim,
            out_dtype(self, dtype),
            widen=widen,
            canonical_bool=canonical_bool,
            acc_dtype=dtype if dtype is not None and not cast_input else None,
        )

    return impl


# --- Group B: INDEX reductions. The traits carry the index in a second accumulator field,
# so these route through the index-aware paths. Ties and NaNs match aten (first wins). ---


def _idx_width(self: torch.Tensor, red: _Red) -> tuple[Any, str]:
    # The in-kernel INDEX accumulator: the winning position ranges over the REDUCED extent, so
    # Int32 overflows at 2**31. The OUTPUT buffer is always int64, aten's index dtype.
    import cutlass

    extent = self.numel() if red is None else math.prod(self.shape[d] for d in red)
    if extent >= _INT32_LIMIT:
        return cutlass.Int64, "i64"
    return cutlass.Int32, "i32"


def _run_arg(
    make_trait: Callable[[Any, Any], Any],
    key: str,
    self: torch.Tensor,
    red: _Red,
    keepdim: bool,
) -> torch.Tensor:
    # The kernel STORES int64 directly, since a trailing widening cast is a whole extra ~2us
    # launch; the accumulator keeps _idx_width's narrower type so shuffles stay cheap.
    if self.numel() == 0:
        # arg* has no identity, so _empty_ok only let this through for an EMPTY output.
        return _empty_result(self, red, keepdim, [torch.int64], [None])[0]
    x = _kernel_input(self)
    acc, _ = _acc_for(x, widen=False)
    idx_cute, tag = _idx_width(self, red)
    trait = make_trait(acc, idx_cute)
    key = key + tag  # distinct idx width -> distinct compiled kernel
    if red is None:
        # The flat index is in logical row-major order, so materialize a
        # non-contiguous input just as ATen does for argmax(dim=None).
        out = kg.reduce_all(trait, key, x.reshape(-1), torch.int64, order="unordered")
    else:
        out = kg.reduce_dim(trait, key, x, sorted(red), torch.int64, order="unordered")
    return _keepdim_reshape(out, self.shape, red, keepdim)


def _run_dim2(
    make_trait: Callable[[Any, Any], Any],
    key: str,
    self: torch.Tensor,
    dim: _Dim,
    keepdim: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Values keep the input dtype; the index is stored as int64 directly, as in _run_arg. Via
    # _normalize_dims, not `dim % self.dim()`, which is undefined for a 0-dim input.
    red = _normalize_dims(dim, self.dim())
    if self.numel() == 0:  # empty output only, as in _run_arg
        vals, idxs = _empty_result(
            self, red, keepdim, [self.dtype, torch.int64], [None, None]
        )
        return vals, idxs
    # Indexed bool traits canonicalize each byte before tie-breaking.
    x = _kernel_input(self, canonical_bool=False)
    acc, kout = _acc_for(x, widen=False)
    idx_cute, tag = _idx_width(self, red)
    trait = make_trait(acc, idx_cute)
    dims = None if red is None else sorted(red)
    vdt = self.dtype if self.dtype in _STORABLE_DTYPES else kout
    vals, idxs = kg.reduce_dim2(
        trait,
        key + tag,
        x,
        dims,
        [vdt, torch.int64],
        order="unordered",
    )
    vals = _keepdim_reshape(vals, self.shape, red, keepdim)
    idxs = _keepdim_reshape(idxs, self.shape, red, keepdim)
    return vals.to(self.dtype), idxs


def _make_index_impl(
    trait_type: Callable[[], Any], key: str, *, with_value: bool = False
) -> Callable[..., _Result]:
    def impl(self: torch.Tensor, dim: _Dim = None, keepdim: bool = False) -> _Result:
        make_trait = lambda acc, idx: trait_type()(acc=acc, idx=idx)  # noqa: E731
        if with_value:
            return _run_dim2(make_trait, key, self, dim, keepdim)
        return _run_arg(
            make_trait,
            key,
            self,
            _normalize_dims(dim, self.dim()),
            keepdim,
        )

    return impl


_argmax_impl = _make_index_impl(lambda: T.ArgMaxOps, "argmax")
_argmin_impl = _make_index_impl(lambda: T.ArgMinOps, "argmin")
_max_dim_impl = _make_index_impl(lambda: T.MaxDimOps, "max.dim", with_value=True)
_min_dim_impl = _make_index_impl(lambda: T.MinDimOps, "min.dim", with_value=True)


# --- Group C: parameterized or non-value single-output reductions. ---


def _correction(correction: Any) -> Any:
    return 1 if correction is None else correction


def _dof_supported(correction: Any) -> bool:
    try:
        float(_correction(correction))
    except (OverflowError, TypeError, ValueError):
        return False
    return True


def _reduced_count(self: torch.Tensor, dim: _Dim) -> int:
    red = _normalize_dims(dim, self.dim())
    return self.numel() if red is None else math.prod(self.shape[d] for d in red)


def _warn_invalid_dof(
    op_name: str, self: torch.Tensor, dim: _Dim, correction: Any
) -> None:
    if not float(_correction(correction)) >= _reduced_count(self, dim):
        return
    warnings.warn(
        f"{op_name}(): degrees of freedom is <= 0. Correction should be strictly "
        "less than the reduction factor (input numel divided by output numel).",
        UserWarning,
        stacklevel=9,
    )


def _run_welford(
    key: str,
    op_name: str,
    self: torch.Tensor,
    dim: _Dim,
    correction: Any,
    keepdim: bool,
    *,
    take_sqrt: bool,
    with_mean: bool,
    out_dtypes: Sequence[torch.dtype] | None = None,
) -> _Result:
    c = _correction(correction)
    _warn_invalid_dof(op_name, self, dim, c)
    red = _normalize_dims(dim, self.dim())
    if self.numel() == 0:
        dtypes = (
            [self.dtype] * (2 if with_mean else 1)
            if out_dtypes is None
            else list(out_dtypes)
        )
        result = _empty_result(self, red, keepdim, dtypes, [float("nan")] * len(dtypes))
        return result if with_mean else result[0]
    trait = T.VarMeanOps if with_mean else T.WelfordOps
    make_trait = lambda acc: trait(  # noqa: E731
        correction=c, take_sqrt=take_sqrt, acc=acc
    )
    # Correction is a const_expr, so it must distinguish compiled kernels.
    if with_mean:
        return _run_dim2_vals(
            make_trait,
            f"{key}mean{c}",
            self,
            dim,
            keepdim,
            out_dtypes=out_dtypes,
        )
    return _run1(
        make_trait,
        f"{key}{c}",
        self,
        red,
        keepdim,
        self.dtype if out_dtypes is None else out_dtypes[0],
    )


def _make_welford_impl(
    key: str,
    *,
    op_name: str | None = None,
    take_sqrt: bool = False,
    with_mean: bool = False,
) -> Callable[..., _Result]:
    def impl(
        self: torch.Tensor,
        dim: _Dim = None,
        *,
        correction: Any = None,
        keepdim: bool = False,
    ) -> _Result:
        return _run_welford(
            key,
            key if op_name is None else op_name,
            self,
            dim,
            correction,
            keepdim,
            take_sqrt=take_sqrt,
            with_mean=with_mean,
        )

    return impl


# linalg_vector_norm ord -> trait factory.
def _norm_trait(ord_val: Any) -> Callable[[Any], Any]:
    if ord_val == float("inf"):
        return lambda acc: T.AbsMaxOps(acc=acc)
    if ord_val == float("-inf"):
        return lambda acc: T.AbsMinOps(acc=acc)
    if ord_val == 0:
        return lambda acc: T.CountNonzeroOps(acc=acc)
    return lambda acc: T.NormOps(float(ord_val), acc=acc)


def _vector_norm_impl(
    self: torch.Tensor,
    ord: Any = 2,
    dim: _Dim = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x = self if dtype is None or dtype is self.dtype else self.to(dtype)
    red = _normalize_dims(dim, x.dim())
    odt = _out_dtype(self, dtype)
    return _run1(_norm_trait(ord), f"vnorm{ord}", x, red, keepdim, odt)


def _flag_out_dtype(
    self: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.dtype:
    # all/any preserve uint8 under ATen's legacy ByteTensor rule.
    return torch.uint8 if self.dtype is torch.uint8 else torch.bool


_sum_impl = _make_impl(lambda: T.SumOps, "sum", _sum_out_dtype)
_mean_impl = _make_impl(lambda: T.MeanOps, "mean", _out_dtype)
_nansum_impl = _make_impl(
    lambda: T.NanSumOps, "nansum", _sum_out_dtype, cast_input=False
)
_prod_impl = _make_impl(lambda: T.ProdOps, "prod", _sum_out_dtype)
_amax_impl = _make_impl(
    lambda: T.AMaxOps, "amax", _out_dtype, widen=False, canonical_bool=False
)
_amin_impl = _make_impl(
    lambda: T.AMinOps, "amin", _out_dtype, widen=False, canonical_bool=False
)
_all_impl = _make_impl(
    lambda: T.AllOps, "all", _flag_out_dtype, widen=False, canonical_bool=False
)
_any_impl = _make_impl(
    lambda: T.AnyOps, "any", _flag_out_dtype, widen=False, canonical_bool=False
)
_count_nonzero_impl = _make_impl(
    lambda: T.CountNonzeroOps,
    "cnz",
    lambda self, dtype: torch.int64,
    canonical_bool=False,
)


def _vector_norm_cond(
    self: torch.Tensor,
    ord: Any = 2,
    dim: _Dim = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> bool:
    return _base_cond(self, dim, key=f"vnorm{ord}") and _out_ok(f"vnorm{ord}", dtype)


# --- Group D: two-output VALUE reductions (var_mean / std_mean / aminmax). Same
# reduce_dim2 path as max.dim, but both outputs cast to the input dtype, not int64. ---


def _run_dim2_vals(
    make_trait: Callable[[Any], Any],
    key: str,
    self: torch.Tensor,
    dim: _Dim,
    keepdim: bool,
    *,
    out_dtypes: Sequence[torch.dtype] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    red = _normalize_dims(dim, self.dim())
    dtypes = [self.dtype] * 2 if out_dtypes is None else list(out_dtypes)
    if self.numel() == 0:
        # Welford handles its NaN identity before reaching this generic pair path, so
        # this is aminmax with an empty output.
        o0, o1 = _empty_result(self, red, keepdim, dtypes, [None, None])
        return o0, o1
    # aminmax can consume bool storage bytes directly; its outputs cast back to bool.
    x = _kernel_input(self, canonical_bool=False)
    acc, kout = _acc_for(x, widen=False)
    dims = None if red is None else sorted(red)
    kernel_dtypes = [d if d in _STORABLE_DTYPES else kout for d in dtypes]
    trait = make_trait(acc)
    if red is None:
        o0, o1 = kg.reduce_all2(trait, key, x, kernel_dtypes)
    else:
        o0, o1 = kg.reduce_dim2(trait, key, x, dims, kernel_dtypes)
    o0 = _keepdim_reshape(o0, self.shape, red, keepdim)
    o1 = _keepdim_reshape(o1, self.shape, red, keepdim)
    return o0.to(dtypes[0]), o1.to(dtypes[1])


_var_impl = _make_welford_impl("var")
_std_impl = _make_welford_impl("std", take_sqrt=True)
_var_mean_impl = _make_welford_impl("var", op_name="var_mean", with_mean=True)
_std_mean_impl = _make_welford_impl(
    "std", op_name="std_mean", take_sqrt=True, with_mean=True
)


def _aminmax_impl(
    self: torch.Tensor, *, dim: int | None = None, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_dim2_vals(
        lambda acc: T.AMinMaxOps(acc=acc), "aminmax", self, dim, keepdim
    )


# --- Full-reduce overloads. ---


def _flagged_elements(self: torch.Tensor) -> torch.Tensor:
    # all.dims / any.dims are the ONE place an empty dim list means reduce NOTHING, not
    # reduce-all: the result is the elementwise nonzero test at the input's own shape.
    return torch.ne(self, 0).to(_flag_out_dtype(self))


def _all_dims_impl(
    self: torch.Tensor,
    dim: Sequence[int] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if dim is not None and len(dim) == 0:
        return _flagged_elements(self)
    return _all_impl(self, dim, keepdim)


def _any_dims_impl(
    self: torch.Tensor,
    dim: Sequence[int] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if dim is not None and len(dim) == 0:
        return _flagged_elements(self)
    return _any_impl(self, dim, keepdim)


def _result_shape(
    self: torch.Tensor,
    dim: _Dim,
    keepdim: bool,
    *,
    empty_dims_keep_input: bool = False,
) -> tuple[int, ...]:
    if (
        empty_dims_keep_input
        and dim is not None
        and not isinstance(dim, int)
        and len(dim) == 0
    ):
        return tuple(self.shape)
    red = _normalize_dims(dim, self.dim())
    return tuple(_reduced_shape(self.shape, red, keepdim))


def _outputs_ok(
    self: torch.Tensor,
    outs: Sequence[torch.Tensor],
    shape: Sequence[int],
    dtypes: Sequence[Sequence[torch.dtype]],
) -> bool:
    if len(outs) != len(dtypes):
        return False
    for out, allowed in zip(outs, dtypes):
        if (
            cap.is_traced(out)
            or out.layout is not torch.strided
            or out.device != self.device
            or out.dtype not in allowed
            or out.is_neg()
            or out.is_conj()
        ):
            return False
        shape_ok = tuple(out.shape) == tuple(shape) or (
            out._base is None and out.untyped_storage().resizable()
        )
        if (
            not shape_ok
            or torch._debug_has_internal_overlap(out) != 0
            or torch._C._overlaps(self, out)
        ):
            return False
    return not (len(outs) == 2 and torch._C._overlaps(outs[0], outs[1]))


def _copy_results(
    result: torch.Tensor | Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    results = (result,) if isinstance(result, torch.Tensor) else tuple(result)
    for out, value in zip(outs, results):
        if out.shape != value.shape:
            if out.numel() != 0:
                warnings.warn(
                    "An output with one or more elements was resized since it had "
                    f"shape {list(out.shape)}, which does not match the required "
                    f"output shape {list(value.shape)}. This behavior is deprecated, "
                    "and in a future PyTorch release outputs will not be resized unless "
                    "they have zero elements. You can explicitly reuse an out tensor t "
                    "by resizing it, inplace, to zero elements with t.resize_(0).",
                    UserWarning,
                    stacklevel=8,
                )
            out.resize_(value.shape)
        out.copy_(value)
    return outs[0] if len(outs) == 1 else tuple(outs)


def _make_dtype_out_cond(
    key: str, *, nansum_rules: bool = False
) -> Callable[..., bool]:
    def cond(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
        out: torch.Tensor,
    ) -> bool:
        if dtype is not None and out.dtype is not dtype:
            return False
        if (
            nansum_rules
            and dtype is None
            and self.dtype.is_floating_point
            and not out.dtype.is_floating_point
        ):
            return False
        effective_dtype = out.dtype if dtype is None else dtype
        return (
            _base_cond(self, dim, key)
            and _out_ok(key, effective_dtype)
            and _outputs_ok(
                self,
                [out],
                _result_shape(self, dim, keepdim),
                [(effective_dtype,)],
            )
        )

    return cond


def _make_out_cond(
    key: str,
    names: tuple[str, ...],
    dtypes: Callable[[torch.Tensor], Sequence[Sequence[torch.dtype]]],
    *,
    empty_dims_keep_input: bool = False,
) -> Callable[..., bool]:
    def cond(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        **kwargs: torch.Tensor,
    ) -> bool:
        outs = [kwargs[name] for name in names]
        return _base_cond(self, dim, key) and _outputs_ok(
            self,
            outs,
            _result_shape(
                self,
                dim,
                keepdim,
                empty_dims_keep_input=empty_dims_keep_input,
            ),
            dtypes(self),
        )

    return cond


def _make_out_impl(
    impl: Callable[..., Any],
    names: tuple[str, ...] = ("out",),
    *,
    out_sets_dtype: bool = False,
) -> Callable[..., _Result]:
    def out_impl(
        self: torch.Tensor,
        dim: _Dim = None,
        keepdim: bool = False,
        **kwargs: Any,
    ) -> _Result:
        outs = [kwargs.pop(name) for name in names]
        if out_sets_dtype and kwargs.get("dtype") is None:
            kwargs["dtype"] = outs[0].dtype
        result = impl(self, dim=dim, keepdim=keepdim, **kwargs)
        return _copy_results(result, outs)

    return out_impl


def _make_welford_out_cond(
    key: str,
    names: tuple[str, ...],
    dtypes: Callable[[torch.Tensor], Sequence[Sequence[torch.dtype]]],
) -> Callable[..., bool]:
    def cond(
        self: torch.Tensor,
        dim: _Dim = None,
        *,
        correction: Any = None,
        keepdim: bool = False,
        **kwargs: torch.Tensor,
    ) -> bool:
        outs = [kwargs[name] for name in names]
        return (
            _base_cond(self, dim, key)
            and _dof_supported(correction)
            and _outputs_ok(
                self,
                outs,
                _result_shape(self, dim, keepdim),
                dtypes(self),
            )
        )

    return cond


def _make_welford_out_impl(
    key: str,
    names: tuple[str, ...] = ("out",),
    *,
    op_name: str | None = None,
    take_sqrt: bool = False,
) -> Callable[..., _Result]:
    def out_impl(
        self: torch.Tensor,
        dim: _Dim = None,
        *,
        correction: Any = None,
        keepdim: bool = False,
        **kwargs: torch.Tensor,
    ) -> _Result:
        outs = [kwargs[name] for name in names]
        x = (
            self
            if len(outs) == 2 or outs[0].dtype is self.dtype
            else self.to(outs[0].dtype)
        )
        result = _run_welford(
            key,
            key if op_name is None else op_name,
            x,
            dim,
            correction,
            keepdim,
            take_sqrt=take_sqrt,
            with_mean=len(outs) == 2,
            out_dtypes=[out.dtype for out in outs],
        )
        return _copy_results(result, outs)

    return out_impl


def _vector_norm_out_cond(
    self: torch.Tensor,
    ord: Any = 2,
    dim: _Dim = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: torch.Tensor,
) -> bool:
    result_dtype = _out_dtype(self, dtype)
    return _vector_norm_cond(self, ord, dim, keepdim, dtype=dtype) and _outputs_ok(
        self,
        [out],
        _result_shape(self, dim, keepdim),
        [(result_dtype,)],
    )


def _vector_norm_out_impl(
    self: torch.Tensor,
    ord: Any = 2,
    dim: _Dim = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: torch.Tensor,
) -> _Result:
    result = _vector_norm_impl(self, ord, dim, keepdim, dtype=dtype)
    return _copy_results(result, [out])


def _same_dtype(self: torch.Tensor) -> tuple[tuple[torch.dtype]]:
    return ((self.dtype,),)


def _long_dtype(self: torch.Tensor) -> tuple[tuple[torch.dtype]]:
    return ((torch.int64,),)


def _flag_dtypes(
    self: torch.Tensor,
) -> tuple[tuple[torch.dtype, torch.dtype]]:
    return ((torch.bool, torch.uint8),)


def _float_dtypes(
    self: torch.Tensor,
) -> tuple[tuple[torch.dtype, ...]]:
    return (_SUPPORTED_DTYPES,)


def _value_index_dtypes(
    self: torch.Tensor,
) -> tuple[tuple[torch.dtype], tuple[torch.dtype]]:
    return (self.dtype,), (torch.int64,)


def _same_pair_dtypes(
    self: torch.Tensor,
) -> tuple[tuple[torch.dtype], tuple[torch.dtype]]:
    return (self.dtype,), (self.dtype,)


def register_reduction_overrides() -> None:
    # CUDA overrides; cu.register_op_override short-circuits when the CuteDSL
    # runtime is unavailable, so this is safe to call unconditionally at import.
    overrides = (
        ("sum.dim_IntList", _make_dtype_cond("sum"), _sum_impl),
        ("mean.dim", _make_dtype_cond("mean"), _mean_impl),
        ("nansum", _make_dtype_cond("nansum"), _nansum_impl),
        ("amax", _make_cond("amax"), _amax_impl),
        ("amin", _make_cond("amin"), _amin_impl),
        ("prod.dim_int", _make_dtype_cond("prod"), _prod_impl),
        ("argmax", _make_cond("argmax"), _argmax_impl),
        ("argmin", _make_cond("argmin"), _argmin_impl),
        ("max.dim", _make_cond("max_dim"), _max_dim_impl),
        ("min.dim", _make_cond("min_dim"), _min_dim_impl),
        ("var.correction", _make_dof_cond("var"), _var_impl),
        ("std.correction", _make_dof_cond("std"), _std_impl),
        ("linalg_vector_norm", _vector_norm_cond, _vector_norm_impl),
        ("all.dim", _make_cond("all"), _all_impl),
        ("any.dim", _make_cond("any"), _any_impl),
        ("count_nonzero.dim_IntList", _make_cond("cnz"), _count_nonzero_impl),
        ("var_mean.correction", _make_dof_cond("var_mean"), _var_mean_impl),
        ("std_mean.correction", _make_dof_cond("std_mean"), _std_mean_impl),
        ("aminmax", _make_cond("aminmax"), _aminmax_impl),
        ("prod", _make_dtype_cond("prod"), _prod_impl),
        ("max", _make_cond("amax"), _amax_impl),
        ("min", _make_cond("amin"), _amin_impl),
        ("all", _make_cond("all"), _all_impl),
        ("any", _make_cond("any"), _any_impl),
        ("all.dims", _make_cond("all"), _all_dims_impl),
        ("any.dims", _make_cond("any"), _any_dims_impl),
        ("count_nonzero", _make_cond("cnz"), _count_nonzero_impl),
    )

    dtype_outs = (
        ("sum.IntList_out", "sum", _sum_impl),
        ("sum.out", "sum", _sum_impl),
        ("mean.out", "mean", _mean_impl),
        ("mean.dtype_out", "mean", _mean_impl),
        ("nansum.out", "nansum", _nansum_impl),
        ("prod.int_out", "prod", _prod_impl),
        ("prod.out", "prod", _prod_impl),
    )
    overrides += tuple(
        (
            op,
            _make_dtype_out_cond(key, nansum_rules=key == "nansum"),
            _make_out_impl(impl, out_sets_dtype=True),
        )
        for op, key, impl in dtype_outs
    )

    # op, trait key, output kwarg names, accepted dtypes, implementation, [] semantics
    fixed_outs = (
        ("amax.out", "amax", ("out",), _same_dtype, _amax_impl, False),
        ("amin.out", "amin", ("out",), _same_dtype, _amin_impl, False),
        ("argmax.out", "argmax", ("out",), _long_dtype, _argmax_impl, False),
        ("argmin.out", "argmin", ("out",), _long_dtype, _argmin_impl, False),
        (
            "max.dim_max",
            "max_dim",
            ("max", "max_values"),
            _value_index_dtypes,
            _max_dim_impl,
            False,
        ),
        ("max.unary_out", "amax", ("out",), _same_dtype, _amax_impl, False),
        (
            "min.dim_min",
            "min_dim",
            ("min", "min_indices"),
            _value_index_dtypes,
            _min_dim_impl,
            False,
        ),
        ("min.unary_out", "amin", ("out",), _same_dtype, _amin_impl, False),
        ("all.out", "all", ("out",), _flag_dtypes, _all_impl, False),
        ("all.dims_out", "all", ("out",), _flag_dtypes, _all_dims_impl, True),
        ("all.all_out", "all", ("out",), _flag_dtypes, _all_impl, False),
        ("any.out", "any", ("out",), _flag_dtypes, _any_impl, False),
        ("any.dims_out", "any", ("out",), _flag_dtypes, _any_dims_impl, True),
        ("any.all_out", "any", ("out",), _flag_dtypes, _any_impl, False),
        (
            "count_nonzero.dim_IntList_out",
            "cnz",
            ("out",),
            _long_dtype,
            _count_nonzero_impl,
            False,
        ),
        (
            "count_nonzero.out",
            "cnz",
            ("out",),
            _long_dtype,
            _count_nonzero_impl,
            False,
        ),
        (
            "aminmax.out",
            "aminmax",
            ("min", "max"),
            _same_pair_dtypes,
            _aminmax_impl,
            False,
        ),
    )
    overrides += tuple(
        (
            op,
            _make_out_cond(
                key, names, dtypes, empty_dims_keep_input=empty_dims_keep_input
            ),
            _make_out_impl(impl, names),
        )
        for op, key, names, dtypes, impl, empty_dims_keep_input in fixed_outs
    )

    welford_outs = (
        ("var.correction_out", "var", "var", ("out",), _float_dtypes, False),
        ("std.correction_out", "std", "std", ("out",), _float_dtypes, True),
        (
            "var_mean.correction_out",
            "var_mean",
            "var",
            ("out0", "out1"),
            _same_pair_dtypes,
            False,
        ),
        (
            "std_mean.correction_out",
            "std_mean",
            "std",
            ("out0", "out1"),
            _same_pair_dtypes,
            True,
        ),
    )
    overrides += tuple(
        (
            op,
            _make_welford_out_cond(cond_key, names, dtypes),
            _make_welford_out_impl(
                impl_key,
                names,
                op_name=cond_key,
                take_sqrt=take_sqrt,
            ),
        )
        for op, cond_key, impl_key, names, dtypes, take_sqrt in welford_outs
    )
    overrides += (
        ("linalg_vector_norm.out", _vector_norm_out_cond, _vector_norm_out_impl),
    )

    for op, cond, impl in overrides:
        cu.register_op_override(
            "aten",
            op,
            "CUDA",
            cond=cond,
            impl=cap.cuda_device_guard(cap.resolve_neg_view(impl)),
        )
