"""aten reduction overrides backed by the CuteDSL reduction kernels.

Registers CUDA overrides for a small set of reduction ops via the _native
registry. Each override is a (cond, impl) pair: ``cond`` decides per-call whether
we should serve the call; when it returns False the dispatcher falls back to the
normal aten kernel. ``cond`` gates on capability (device, arch, dtype, contiguity,
valid dim) AND on whether one of our FAST kernels (row K1/xcta, col K2) can serve
the post-TI-coalesce geometry (see ``_geometry_supported`` / kernel_general
``fast_kind``). The general K0 kernel is correct for any geometry but ~5-8x slower
than aten, so a K0-only geometry is DECLINED to aten rather than served -- gating
here IS a performance decision, because running K0 would regress vs the very kernel
we fall back to. Host-bound cases on the served (fast) regimes are addressed
elsewhere (CUDA-graph capture, dynamic-M).

The single-output value/sum reductions are wired here: ``sum`` and ``mean``
(fp32-accumulated, optional out ``dtype``) plus ``amax`` / ``amin`` / ``prod``
(output dtype follows the input). The remaining reductions (var/std, norm, arg*,
*_mean, aminmax, all/any, count_nonzero) carry an index or a second output and
follow the same (cond, impl) template; they are added incrementally as the
trait/kernel support for those shapes lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule


if TYPE_CHECKING:
    from .._cutedsl import traits as T
    from . import kernel_general as kg
else:
    # T (trait library) and kg (general-kernel dispatcher) import `cutlass`,
    # which `import torch` must not do (the lazy-DSL-import contract; see
    # test_no_dsl_imports_after_import_torch). Neither is touched by a `cond`
    # or by registration -- only by the `*_impl` functions on a real
    # (non-declined) call, where the DSL runtime is present. Lazy module
    # proxies keep the `T.SumOps` / `kg.reduce_dim` call sites unchanged.
    T = LazyModule("torch._native.ops._cutedsl.traits")
    kg = LazyModule("torch._native.ops.reductions.kernel_general")


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _acc_policy():
    # Lazy (its values are `cutlass` dtypes): resolved on first real call so
    # `import torch` stays cutlass-free. INPUT torch dtype -> (accumulator cute
    # dtype, torch dtype the kernel writes). Kernel out dtype == the accumulator
    # dtype; the impl casts the result to aten's final output dtype. The one
    # place to extend coverage: add a row plus matching trait/kernel support.
    import cutlass

    return {
        torch.float16: (cutlass.Float32, torch.float32),
        torch.bfloat16: (cutlass.Float32, torch.float32),
        torch.float32: (cutlass.Float32, torch.float32),
    }


def _normalize_dims(dim, ndim: int):
    """aten dim arg (None | int | list[int]) -> sorted set of non-negative axes.

    Returns None for "reduce all" (dim is None or empty list). Caller must have
    already validated the dims via `_dims_ok` (this assumes in-range, so it is
    only ever invoked from `*_impl`, never from `cond`)."""
    if dim is None:
        return None
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) == 0:
        return None
    return {d % ndim for d in dims}


def _dims_ok(dim, ndim: int) -> bool:
    # Decline (-> fall back to aten) any dim arg aten would reject or that our
    # kernels can't serve, WITHOUT raising -- a cond must never throw. Mirrors
    # aten's dim validation just enough to defer: every reduced axis must be in
    # [-ndim, ndim), no duplicates (after normalization), and ndim must be small
    # enough that aten doesn't itself raise the >64-dim error. For a 0-dim
    # (scalar) tensor ndim==0: only dim in {None, [], 0, -1} reduce-all is valid;
    # the modulo math is undefined, so just decline scalars (aten handles them
    # trivially and they are not a perf concern).
    if ndim == 0:
        return False
    if ndim > 64:
        return False
    if dim is None:
        return True
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) == 0:
        return True
    for d in dims:
        if d < -ndim or d >= ndim:  # out of range -> let aten raise
            return False
    norm = [d % ndim for d in dims]
    if len(set(norm)) != len(norm):  # duplicate dims -> let aten raise
        return False
    return True


def _geometry_supported(x: torch.Tensor, dim, nouts=1, has_index=False) -> bool:
    # Gate on whether our FAST kernels (row K1/xcta, col K2) can serve this call --
    # NOT merely whether we are correct. The K0 general kernel is correct for any
    # geometry but ~5-8x slower than aten, so declining a K0-only geometry to aten
    # is a WIN, not a fallback. We classify the POST-TI-coalesce geometry (via
    # kernel_general.fast_kind, the SAME source of truth the router uses) so a
    # contiguous n-D reduction that coalesces to a fast row/col (e.g. (A,B,C) over
    # dim 2) is served, while a genuinely irregular one (transpose, mixed multi-run,
    # gapped) declines to aten. Reduce-all (dim None/empty) is always fast (xcta).
    # K0 needs contiguity for its flat-offset addressing; a non-contiguous input is
    # neither fast-serviceable nor K0-serviceable here, so decline. A COW input no
    # longer needs a cond check: the kernels export it read-only via
    # ReadOnlyTensorWrapper (launch._ro). dim validity is checked by _dims_ok.
    if not (x.is_contiguous() and x.numel() > 0):
        return False
    red = _normalize_dims(dim, x.dim())
    if red is None:  # reduce-all -> xcta / two-stage: always a fast path
        return True
    # kg is the lazy proxy bound at module top; touching it here (in a cond, i.e. a
    # real call) fires the cutlass import, which is fine -- registration never calls
    # this. Same TI decode + fast_kind the router uses, so gate and router agree.
    red_pairs, kept_pairs = kg._ti_pairs(x, kg._probe(x, red))
    return kg.fast_kind(red_pairs, kept_pairs, nouts, has_index) is not None


def _keepdim_reshape(out: torch.Tensor, x_shape, red: set | None, keepdim: bool):
    # Our kernels return the NON-keepdim result. Restore size-1 reduced dims when
    # keepdim=True so the output matches aten's shape exactly.
    if not keepdim:
        return out
    if red is None:
        target = [1] * len(x_shape)
    else:
        target = [1 if i in red else s for i, s in enumerate(x_shape)]
    return out.reshape(target)


# ----------------------------------------------------------- shared cond helpers


def _out_dtype(self: torch.Tensor, dtype) -> torch.dtype:
    # aten output-dtype rule for float reductions: the explicit `dtype` if given,
    # else the input's own dtype (fp16 in -> fp16 out). Our kernels accumulate in
    # fp32 internally and we cast the result to this at the end.
    return dtype if dtype is not None else self.dtype


def _supported_out_dtype(dtype) -> bool:
    # We can deliver any of our supported float out dtypes (we accumulate in fp32
    # and cast down). A non-float requested dtype falls back to aten.
    return dtype is None or dtype in _SUPPORTED_DTYPES


def _base_cond(self, dim, nouts=1, has_index=False) -> bool:
    # Shared capability gate for every reduction override: not traced (fake/meta) +
    # CUDA + sm9/10 + a supported float dtype + current device + valid dim +
    # FAST-serviceable geometry. Per-op conds pass nouts/has_index (they know these
    # statically) so the geometry gate matches the op's actual fast-path options
    # (K2 col is single-output value-only; index/2-out ops route only row/xcta or
    # K0 -> decline). Per-op conds add only their extra checks (e.g. an out dtype=
    # argument). Never raises -- a cond that throws would crash the dispatcher
    # instead of falling back. (COW is no longer gated: inputs export read-only,
    # see launch._ro.)
    # A NEG/CONJ bit declines: it is lazy metadata, so the buffer we would read through
    # dlpack holds the UNNEGATED values. It also must not be resolved inside a cond --
    # aten materializes such a view by CALLING copy_, so any override of copy_ would be
    # re-entered from here. Declining lets aten resolve the bit and re-dispatch the
    # materialized tensor, which this cond then accepts normally.
    #
    # A base pointer that is not 16-byte aligned also declines. The row/col/xcta wraps
    # claim an alignment derived from N and the vector width, and from_dlpack VALIDATES
    # that claim -- a VIEW into a larger tensor need not honour it (base[1:] on fp64
    # starts 8 B in), which raised "Misaligned Tensor data" mid-call on an ordinary
    # sum/amax over a slice. Clamping the claim per call is not enough: the compiled
    # kernel bakes its load width, so a plan built for an aligned call cannot serve a
    # misaligned one. Fresh allocations are >=256 B aligned, so this only sheds sliced
    # views, which aten handles correctly.
    return (
        not cap.is_traced(self)
        and cap.device_ok(self)
        and self.dtype in _SUPPORTED_DTYPES
        and cap.on_current_device(self)
        and not self.is_neg()
        and not self.is_conj()
        and self.const_data_ptr() % 16 == 0
        and _dims_ok(dim, self.dim())
        and _geometry_supported(self, dim, nouts, has_index)
    )


def _run1(make_trait, key, self, red, keepdim, out_torch_dtype):
    # Run a single-output trait through the dispatcher (TI-driven general path +
    # fast paths) and restore keepdim. acc/kernel-out come from _ACC_POLICY; the
    # final result is cast to `out_torch_dtype` (aten's output dtype for this op).
    # Every single-output reduction override funnels through here.
    acc, kout = _acc_policy()[self.dtype]
    trait = make_trait(acc)
    if red is None:
        out = kg.reduce_all(trait, key, self, kout)
    else:
        out = kg.reduce_dim(trait, key, self, sorted(red), kout)
    out = _keepdim_reshape(out, self.shape, red, keepdim)
    return out.to(out_torch_dtype)


# --- sum / mean: fp32-accumulated, optional out dtype= (else input dtype). ---


def _sum_cond(self, dim=None, keepdim=False, *, dtype=None):
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


def _sum_impl(self, dim=None, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.SumOps(acc=acc), "sum", self, red, keepdim, odt)


def _mean_cond(self, dim=None, keepdim=False, *, dtype=None):
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


def _mean_impl(self, dim=None, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.MeanOps(acc=acc), "mean", self, red, keepdim, odt)


# --- amax / amin / prod: single-output VALUE reductions; output dtype follows the
# input (amax/amin) or the optional out dtype= (prod). ---


def _amax_impl(self, dim=(), keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AMaxOps(acc=acc), "amax", self, red, keepdim, self.dtype)


def _amin_impl(self, dim=(), keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AMinOps(acc=acc), "amin", self, red, keepdim, self.dtype)


def _prod_impl(self, dim, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.ProdOps(acc=acc), "prod", self, red, keepdim, odt)


def _amax_cond(self, dim=(), keepdim=False):
    return _base_cond(self, dim)


def _amin_cond(self, dim=(), keepdim=False):
    return _base_cond(self, dim)


def _prod_cond(self, dim, keepdim=False, *, dtype=None):
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


# --- Group B: INDEX reductions. argmax / argmin return the winning index (int64);
# max.dim / min.dim return (values, indices). The traits carry the index in a
# second accumulator field (has_index), so these route through the index-aware
# dispatcher paths (two-stage K0 / reduce_dim2) rather than the value fast paths.
# Tie-break and NaN handling match aten (first max/min, first NaN). ---


def _idx_width(self, red):
    # Choose the in-kernel INDEX dtype for an argmax/argmin/max.dim/min.dim: the
    # winning position ranges over the REDUCED extent, so Int32 overflows once that
    # reaches 2^31. Return (cute idx dtype, torch kernel-out dtype, key tag). Int32 is
    # the default (cheaper partials + shuffle); Int64 only when needed. red is the set
    # of reduced axes (None = reduce-all).
    import cutlass

    if red is None:
        extent = self.numel()
    else:
        extent = 1
        for d in red:
            extent *= self.shape[d]
    if extent > (1 << 31) - 1:
        return cutlass.Int64, torch.int64, "i64"
    return cutlass.Int32, torch.int32, "i32"


def _run_arg(make_trait, key, self, red, keepdim):
    # Single-output INDEX reduction (argmax/argmin). aten indices are int64; the kernel
    # emits its index field (Int32 or, for a huge reduced extent, Int64 -- see
    # _idx_width) and we cast the result up to int64. acc dtype is the value
    # accumulator (from _ACC_POLICY).
    acc, _ = _acc_policy()[self.dtype]
    idx_cute, idx_torch, tag = _idx_width(self, red)
    trait = make_trait(acc, idx_cute)
    key = key + tag  # distinct idx width -> distinct compiled kernel
    if red is None:
        out = kg.reduce_all(trait, key, self, idx_torch)
    else:
        out = kg.reduce_dim(trait, key, self, sorted(red), idx_torch)
    out = _keepdim_reshape(out, self.shape, red, keepdim)
    return out.to(torch.int64)


def _run_dim2(make_trait, key, self, dim, keepdim):
    # Two-output reduction over a single dim (max.dim/min.dim): (values, indices).
    # Values keep the input dtype, indices are int64 (kernel emits Int32/Int64 per
    # _idx_width -- Int64 when the reduced dim can exceed the Int32 range).
    acc, kout = _acc_policy()[self.dtype]
    red = {dim % self.dim()}
    idx_cute, idx_torch, tag = _idx_width(self, red)
    vals, idxs = kg.reduce_dim2(
        make_trait(acc, idx_cute), key + tag, self, sorted(red), [kout, idx_torch]
    )
    vals = _keepdim_reshape(vals, self.shape, red, keepdim).to(self.dtype)
    idxs = _keepdim_reshape(idxs, self.shape, red, keepdim).to(torch.int64)
    return vals, idxs


def _argmax_impl(self, dim=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run_arg(
        lambda acc, idx: T.ArgMaxOps(acc=acc, idx=idx), "argmax", self, red, keepdim
    )


def _argmin_impl(self, dim=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run_arg(
        lambda acc, idx: T.ArgMinOps(acc=acc, idx=idx), "argmin", self, red, keepdim
    )


def _max_dim_impl(self, dim, keepdim=False):
    return _run_dim2(
        lambda acc, idx: T.MaxDimOps(acc=acc, idx=idx), "max.dim", self, dim, keepdim
    )


def _min_dim_impl(self, dim, keepdim=False):
    return _run_dim2(
        lambda acc, idx: T.MinDimOps(acc=acc, idx=idx), "min.dim", self, dim, keepdim
    )


def _argmax_cond(self, dim=None, keepdim=False):
    return _base_cond(self, dim, has_index=True)


def _argmin_cond(self, dim=None, keepdim=False):
    return _base_cond(self, dim, has_index=True)


def _max_dim_cond(self, dim, keepdim=False):
    # max.dim/min.dim take a required single int dim (not a list); _base_cond's
    # _dims_ok accepts the int form and declines scalars / out-of-range.
    return _base_cond(self, dim, nouts=2, has_index=True)


def _min_dim_cond(self, dim, keepdim=False):
    return _base_cond(self, dim, nouts=2, has_index=True)


# --- Group C: single-output reductions with a parameter or a non-float output.
# var/std (Welford + correction; std = sqrt(var)), linalg_vector_norm (ord selects
# the trait), all/any (bool output), count_nonzero (int64 output). All accumulate in
# the float _ACC_POLICY acc and cast the projected value to aten's output dtype. ---


def _correction(correction, unbiased=None):
    # aten var/std correction: explicit `correction` wins; else `unbiased` (the old
    # bool API) maps True->1 / False->0; default unbiased -> 1.
    if correction is not None:
        return correction
    if unbiased is None:
        return 1
    return 1 if unbiased else 0


def _var_impl(self, dim=None, *, correction=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    c = _correction(correction)
    # `correction` is baked into the kernel as a const_expr, so it MUST be in the
    # trait_key -- else the first-compiled correction is reused for all others.
    return _run1(
        lambda acc: T.WelfordOps(correction=c, acc=acc),
        f"var{c}",
        self,
        red,
        keepdim,
        self.dtype,
    )


def _std_impl(self, dim=None, *, correction=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    c = _correction(correction)
    return _run1(
        lambda acc: T.WelfordOps(correction=c, take_sqrt=True, acc=acc),
        f"std{c}",
        self,
        red,
        keepdim,
        self.dtype,
    )


# linalg_vector_norm ord -> trait factory. inf/-inf are max/min |x| (AbsMax/AbsMin);
# finite p uses NormOps(p). ord=0 (count nonzero) is NOT handled here -> falls back.
def _norm_trait(ord_val):
    if ord_val == float("inf"):
        return lambda acc: T.AbsMaxOps(acc=acc)
    if ord_val == float("-inf"):
        return lambda acc: T.AbsMinOps(acc=acc)
    return lambda acc: T.NormOps(float(ord_val), acc=acc)


def _vector_norm_impl(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(_norm_trait(ord), f"vnorm{ord}", self, red, keepdim, odt)


def _all_impl(self, dim, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AllOps(acc=acc), "all", self, red, keepdim, torch.bool)


def _any_impl(self, dim, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AnyOps(acc=acc), "any", self, red, keepdim, torch.bool)


def _count_nonzero_impl(self, dim):
    red = _normalize_dims(dim, self.dim())
    return _run1(
        lambda acc: T.CountNonzeroOps(acc=acc), "cnz", self, red, False, torch.int64
    )


def _var_cond(self, dim=None, *, correction=None, keepdim=False):
    return _base_cond(self, dim)


def _std_cond(self, dim=None, *, correction=None, keepdim=False):
    return _base_cond(self, dim)


def _vector_norm_cond(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    # ord=0 is a nonzero-count, not a |x|**p norm -> let aten handle it.
    return _base_cond(self, dim) and ord != 0 and _supported_out_dtype(dtype)


def _all_cond(self, dim, keepdim=False):
    return _base_cond(self, dim)


def _any_cond(self, dim, keepdim=False):
    return _base_cond(self, dim)


def _count_nonzero_cond(self, dim):
    return _base_cond(self, dim)


def register_reduction_overrides() -> None:
    # CUDA overrides; cu.register_op_override short-circuits when the CuteDSL
    # runtime is unavailable, so this is safe to call unconditionally at import.
    cu.register_op_override(
        "aten", "sum.dim_IntList", "CUDA", cond=_sum_cond, impl=_sum_impl
    )
    cu.register_op_override(
        "aten", "mean.dim", "CUDA", cond=_mean_cond, impl=_mean_impl
    )
    # Group A: amax / amin / prod (single-output value reductions).
    cu.register_op_override("aten", "amax", "CUDA", cond=_amax_cond, impl=_amax_impl)
    cu.register_op_override("aten", "amin", "CUDA", cond=_amin_cond, impl=_amin_impl)
    cu.register_op_override(
        "aten", "prod.dim_int", "CUDA", cond=_prod_cond, impl=_prod_impl
    )
    # Group B: argmax / argmin (int64 index) and max.dim / min.dim (values, indices).
    cu.register_op_override(
        "aten", "argmax", "CUDA", cond=_argmax_cond, impl=_argmax_impl
    )
    cu.register_op_override(
        "aten", "argmin", "CUDA", cond=_argmin_cond, impl=_argmin_impl
    )
    cu.register_op_override(
        "aten", "max.dim", "CUDA", cond=_max_dim_cond, impl=_max_dim_impl
    )
    cu.register_op_override(
        "aten", "min.dim", "CUDA", cond=_min_dim_cond, impl=_min_dim_impl
    )
    # Group C: var / std (correction), linalg_vector_norm (ord), all / any (bool
    # output), count_nonzero (int64 output). Single-output, float input.
    cu.register_op_override(
        "aten", "var.correction", "CUDA", cond=_var_cond, impl=_var_impl
    )
    cu.register_op_override(
        "aten", "std.correction", "CUDA", cond=_std_cond, impl=_std_impl
    )
    cu.register_op_override(
        "aten",
        "linalg_vector_norm",
        "CUDA",
        cond=_vector_norm_cond,
        impl=_vector_norm_impl,
    )
    cu.register_op_override("aten", "all.dim", "CUDA", cond=_all_cond, impl=_all_impl)
    cu.register_op_override("aten", "any.dim", "CUDA", cond=_any_cond, impl=_any_impl)
    cu.register_op_override(
        "aten",
        "count_nonzero.dim_IntList",
        "CUDA",
        cond=_count_nonzero_cond,
        impl=_count_nonzero_impl,
    )
