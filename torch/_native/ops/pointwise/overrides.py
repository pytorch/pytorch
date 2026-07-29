# Generic registration of the pointwise definition table as aten CUDA overrides. Each row -> a
# (cond, impl) pair; the impl picks compute/output dtypes via aten's elementwise type
# promotion, bakes the row's scalar args as compute-dtype constants, and runs the one
# generic elementwise kernel. No per-op code beyond the table row.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from torch._prims_common import elementwise_dtypes

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from .table import (
    _INT_DTYPES,
    POINTWISE_DEF_TABLE,
    POINTWISE_VARIANTS,
    PointwiseDef,
    variant_aten_name,
)


# The launch glue (_L), kernel runner (K), and op-math module (ops) all import
# `cutlass`, which `import torch` must not do (the lazy-DSL-import contract; see
# test_no_dsl_imports_after_import_torch). None is touched by a `cond` or by
# registration -- only by the `*_impl` closures on a real (non-declined) call, where
# the DSL runtime is present. table.py (imported above) is cutlass-free and carries
# the registration metadata; a row's `fn` is a NAME resolved via ops.get_fn() at call
# time. Bind the cutlass-laden modules as lazy proxies so their imports fire then.
if TYPE_CHECKING:
    from .._cutedsl import launch as _L
    from . import kernel as K, ops
else:
    _L = LazyModule("torch._native.ops._cutedsl.launch")
    K = LazyModule("torch._native.ops.pointwise.kernel")
    ops = LazyModule("torch._native.ops.pointwise.ops")


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _layout_ok(t) -> bool:
    # Accept contiguous OR row-dense-gapped inputs. Contiguous -> the flat vec (or
    # strided-broadcast) path; row-dense-gapped ([M,N]:(K,1) with K>N, or an n-D
    # last-dim slice) -> the rowvec path (vectorize within rows, ~2x aten). Genuinely
    # irregular layouts (transpose, channels-last, multi-gap) return False here and
    # decline to aten: our general strided path IS correct for them but ~0.25x aten
    # (aten's magic-divider offset kernel is faster), so declining is the right call --
    # same "correct-but-slower-than-aten -> decline" rule as the reductions' K0 gate.
    if t.is_contiguous():
        return True
    return K._row_gap_view(t) is not None


def _layouts_serveable(ins) -> bool:
    # PERF gate across the operand SET: gapped inputs are only fast when the whole set
    # hits the rowvec path, which needs every operand to share ONE row-gap geometry and
    # every other operand to be absent (unary) -- a mix of gapped + contiguous, or two
    # different gaps, falls to the ~0.25x-aten strided path, so decline those to aten
    # instead (serve-only-when-competitive). All-contiguous always passes (vec/strided-
    # broadcast paths are competitive there).
    gaps = {K._row_gap_view(t) for t in ins if not t.is_contiguous()}
    if not gaps:
        return True  # all contiguous
    if len(gaps) > 1:
        return False  # two different gap geometries -> strided-slow -> decline
    # One gap geometry: every operand must share it (a contiguous operand of the same
    # shape has a DIFFERENT (row_step == run) geometry -> mixed -> strided -> decline).
    return all(K._row_gap_view(t) in gaps for t in ins)


def _supported(t, dtypes) -> bool:
    # COW is not gated: inputs export read-only (launch._ro / ReadOnlyTensorWrapper),
    # so a COW input flows through the kernel without materializing.
    return (
        isinstance(t, torch.Tensor)
        and t.dtype in dtypes
        and _layout_ok(t)
        and t.numel() > 0
        and not cap.is_traced(t)
    )


def _scalars(row: PointwiseDef, args, kwargs):
    # Resolve the row's named scalar args, which the caller may pass either
    # positionally (after the nin tensors) or by keyword (e.g. torch.add(x, y,
    # alpha=2)). Missing -> the row's aten default (row.scalar_defaults, aligned with
    # row.scalars; unset -> 1, the add/sub-alpha convention). Tuple in row.scalars order.
    pos = args[row.nin :]
    out = []
    for i, name in enumerate(row.scalars):
        if name in kwargs:
            out.append(kwargs[name])
        elif i < len(pos):
            out.append(pos[i])
        else:
            out.append(row.scalar_defaults[i] if row.scalar_defaults else 1)
    return tuple(out)


# (promotion kind, input dtypes) -> (compute dtype, functional result dtype). Memoized
# module-wide: promotion is a pure function of these, shape-independent, and BOTH the
# cond (safe-cast gate) and impl (kernel dtypes) need it, so a shared cache avoids the
# ~4us elementwise_dtypes call on the hot path. compute = fp32 opmath dtype; result =
# the functional output dtype (what an alloc/`.out` result would be before any cast).
_promo_cache: dict = {}


def _result_dtypes(row: PointwiseDef, in_dtypes):
    got = _promo_cache.get((row.promotion, in_dtypes))
    if got is None:
        probes = [torch.empty(0, dtype=d, device="cuda") for d in in_dtypes]
        compute, result = elementwise_dtypes(*probes, type_promotion_kind=row.promotion)
        got = (compute, result)
        _promo_cache[(row.promotion, in_dtypes)] = got
    return got


def _out_tensor(variant, row, args, kwargs):
    # The mutated target for a non-functional variant, or None (functional). "out_kw" ->
    # the `out=` kwarg (always keyword in aten's schema); "self" -> operand 0 (in-place).
    if variant.out_from == "out_kw":
        return kwargs.get("out")
    if variant.out_from == "self":
        return args[0] if args else None
    return None


def _make_cond(row: PointwiseDef, variant):
    # Serve when all `nin` operands are supported (dtype/contiguous/nonzero), on one
    # device, scalars non-complex, and the variant's shape rule holds. Broadcasting
    # matches aten: operands carry size-1 dims as stride-0 (strided path), output is
    # contiguous broadcast_shapes(...). Variant-specific gates are DATA-driven off
    # variant.shape_rule / variant.out_from (see table.PointwiseVariant):
    #   free    (functional/.out): operands must be broadcast-compatible.
    #   eq_self (in-place): broadcast_shapes(operands) must EQUAL self.shape -- in-place
    #           cannot grow self (aten raises otherwise); operands broadcast UP to it.
    # For .out/in-place the target tensor must be supported (contiguous, on device) AND
    # the promotion RESULT dtype must be safely castable into the target dtype -- aten
    # rejects an "unsafe cast" (f32 result -> int/bool out=, or int self.add_(float)),
    # so we DECLINE those and let aten raise its exact error. can_cast(result, target)
    # is aten's own rule (verified against eager for both .out and in-place).
    # Accepted INPUT dtypes: the float base (or row.dtypes override) PLUS any integer
    # dtypes the row opts into -- int_dtypes (integer-in/integer-out ops) or _INT_DTYPES
    # when int_via_float (int input promotes to a float result, served by the float path).
    # See table.PointwiseDef. bool stays excluded (no bool arithmetic; neg(bool) raises).
    base = row.dtypes or _SUPPORTED
    ints = row.int_dtypes or (_INT_DTYPES if row.int_via_float else ())
    dtypes = tuple(base) + tuple(ints)

    def cond(*args, **kwargs):
        ins = args[: row.nin]
        if len(ins) < row.nin or not all(_supported(t, dtypes) for t in ins):
            return False
        if not (cap.device_ok(ins[0]) and cap.on_current_device(ins[0])):
            return False
        if any(isinstance(s, complex) for s in _scalars(row, args, kwargs)):
            return False
        if any(t.device != ins[0].device for t in ins[1:]):
            return False
        if not _layouts_serveable(ins):
            return False  # gapped operand set the rowvec path can't serve -> aten
        try:
            bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        except RuntimeError:
            return False  # incompatible -> aten raises the precise size-mismatch error
        tgt = _out_tensor(variant, row, args, kwargs)
        if variant.out_from != "alloc":
            if not (
                isinstance(tgt, torch.Tensor)
                and tgt.is_contiguous()
                and tgt.device == ins[0].device
                and not cap.is_traced(tgt)
            ):
                return False
            # Safe-cast gate: promotion result must fit the target dtype (else aten raises).
            _, result_dtype = _result_dtypes(row, tuple(t.dtype for t in ins))
            if not torch.can_cast(result_dtype, tgt.dtype):
                return False
        if variant.shape_rule == "eq_self":
            # In-place: the broadcast result must fit self exactly (no grow).
            return tuple(bshape) == tuple(tgt.shape)
        return True

    return cond


def _make_impl(row: PointwiseDef, variant):
    def _promo(in_dtypes):
        # compute + functional output dtypes for this input-dtype tuple, off the shared
        # _result_dtypes memo. out_dtypes applies the row's escape hatch (e.g. frexp ->
        # [float, int32]); the common case is [result] * nout.
        compute, result = _result_dtypes(row, in_dtypes)
        out_dtypes = row.out_dtypes(result) if row.out_dtypes else [result] * row.nout
        return _L.torch2cute[compute], compute, tuple(out_dtypes)

    def _run_into(targets, ins, scalars, in_dtypes):
        # THE canonical kernel (mirrors aten's structured .out kernel): compute row.fn
        # over the broadcast of `ins` and write into `targets`, casting each element to
        # the TARGET tensor's own dtype. Every variant funnels through here -- they
        # differ ONLY in how `targets` is chosen (below). out_dtypes = the targets'
        # dtypes (NOT the promotion result): .out keeps the out= tensor's dtype and
        # in-place keeps self's, both of which aten casts the compute result into.
        ct, compute, _ = _promo(in_dtypes)
        consts = tuple(ct(s) for s in scalars)
        out_dtypes = tuple(t.dtype for t in targets)
        # Layout + out dtype are baked into the compiled kernel -> in the cache key.
        # Variant is NOT keyed: two variants with the same operand layout and target
        # dtype compile the identical kernel (the target tensor is a runtime arg), so
        # they correctly share one plan.
        key = (
            row.aten,
            in_dtypes,
            tuple((t.shape, t.stride()) for t in ins),
            out_dtypes,
            scalars,
        )
        # int_fn: the Int-compute variant of the math for ops whose integer semantics
        # differ from float (fmod/remainder/floor_divide -- truncating int division vs
        # cute.math.floor, which rejects Int).
        fn_name = (
            row.int_fn
            if row.int_fn is not None and not compute.is_floating_point
            else row.fn
        )
        K.run(
            ops.get_fn(fn_name),
            key,
            row.nin,
            row.nout,
            consts,
            ct,
            compute,
            ins,
            out_dtypes,
            out=targets,
        )

    def _targets(args, kwargs, ins, in_dtypes):
        # Choose the tensors _run_into writes to, per variant.out_from (DATA):
        #   alloc  (functional): fresh contiguous tensors of the PROMOTION out dtypes,
        #                        broadcast shape.
        #   out_kw (.out): the out= kwarg tensor(s), resized to the broadcast shape to
        #                  match aten (which keeps the out tensor's dtype).
        #   self   (in-place): operand 0 (its shape already == broadcast shape via cond).
        bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        if variant.out_from == "alloc":
            _, _, promo_out = _promo(in_dtypes)
            return [
                torch.empty(bshape, device=ins[0].device, dtype=d) for d in promo_out
            ]
        if variant.out_from == "out_kw":
            out = kwargs["out"]
            tgts = list(out) if isinstance(out, (tuple, list)) else [out]
            for t in tgts:
                if tuple(t.shape) != tuple(bshape):
                    # aten warns when it resizes a NON-EMPTY out tensor (a deprecation);
                    # resizing an empty (numel 0) out is silent. Match that exactly.
                    if t.numel() != 0:
                        warnings.warn(
                            "An output with one or more elements was resized since it "
                            f"had shape {tuple(t.shape)}, which does not match the "
                            f"required output shape {tuple(bshape)}. This behavior is "
                            "deprecated, and in a future PyTorch release outputs will "
                            "not be resized unless they have zero elements. You can "
                            "explicitly reuse an out tensor t by resizing it, inplace, "
                            "to zero elements with t.resize_(0).",
                            UserWarning,
                            stacklevel=2,
                        )
                    t.resize_(bshape)
            return tgts
        return [args[0]]  # self

    def impl(*args, **kwargs):
        ins = list(args[: row.nin])
        in_dtypes = tuple(t.dtype for t in ins)
        scalars = _scalars(row, args, kwargs)
        targets = _targets(args, kwargs, ins, in_dtypes)
        _run_into(targets, ins, scalars, in_dtypes)
        return targets[0] if row.nout == 1 else tuple(targets)

    return impl


# ---------------------------------------------------------------------------
# Conversions: _to_copy (.to(dtype)/.float()/.half()/clone-with-dtype) and copy_.
# These do not fit the PointwiseDef table: there is NO type promotion (the target
# dtype is given explicitly), _to_copy carries layout/device/memory_format kwargs
# that gate serveability rather than math, and copy_ casts ANY dtype pair with no
# safe-cast rule (unlike .out/in-place, where aten raises on an unsafe cast). Both
# funnel into the same K.run kernel as every other pointwise op: the load-side
# packed convert to the COMPUTE dtype (= the target dtype) IS the conversion and
# `fn` is the identity. A bool TARGET instead computes aten's nonzero test in the
# SOURCE dtype (_to_bool); bool->bool is declined (rare, plain aten copy is fine).
# ---------------------------------------------------------------------------

# Dtypes conversions serve, as SOURCE or TARGET: all floats + the full integer
# matrix. Excluded (decline -> aten): complex (no DSL complex), float8_* (the DSL
# conversion lowering ICEs on fp8 today -- retest on a CuteDSL upgrade).
_CONV_DTYPES = (
    _SUPPORTED
    + _INT_DTYPES
    + (
        torch.int8,
        torch.int16,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )
)
_CONV_SRC_DTYPES = _CONV_DTYPES + (torch.bool,)


def _conv_serveable(src, dst_dtype):
    # f64 -> u16 declines: aten saturates finite values (-128.5 -> 0) but sends
    # nan to 0 where the direct convert gives 32768, and the int64 detour would
    # wrap the finite values -- no single compute path matches this hybrid.
    if src.dtype is torch.float64 and dst_dtype is torch.uint16:
        return False
    return (
        _supported(src, _CONV_SRC_DTYPES)
        and dst_dtype in _CONV_DTYPES + (torch.bool,)
        and not (dst_dtype is torch.bool and src.dtype is torch.bool)
        and cap.device_ok(src)
        and cap.on_current_device(src)
    )


def _needs_int64_detour(src_dtype, dst_dtype):
    # Casts whose DIRECT hardware conversion diverges from aten; an int64 compute
    # (load-side convert to Int64, store-side wrap to the target width) matches
    # aten bit-for-bit in both cases:
    #   float -> u8/i8/i16: aten converts through a WIDER integer then truncates
    #     -- so -1.7 -> u8 WRAPS to 255, and inf -> i16 is the low 16 bits of the
    #     i64 saturate (-1) -- where the direct hardware convert saturates at the
    #     target width. u16 is the odd one out: aten SATURATES there (-128.5 -> 0),
    #     matching the direct convert, so it stays direct; 32/64-bit targets match
    #     the direct convert too (including their inf/nan patterns).
    #   signed int -> WIDER unsigned: aten sign-extends then reinterprets
    #     (int16 -50 -> uint32 4294967246); the DSL's direct signed->unsigned
    #     convert wraps at the SOURCE width then zero-extends (-> 65486).
    #     Same-or-narrower unsigned targets truncate identically either way.
    if dst_dtype.is_floating_point or dst_dtype is torch.bool:
        return False
    if src_dtype.is_floating_point:
        return dst_dtype in (torch.uint8, torch.int8, torch.int16)
    if dst_dtype.is_signed or not src_dtype.is_signed:
        return False
    return torch.iinfo(dst_dtype).bits > torch.iinfo(src_dtype).bits


def _run_conversion(aten_name, src, target):
    # One-input identity/nonzero kernel writing into `target` (its dtype is the
    # cast). compute = target dtype for a numeric cast (the load converts, the
    # store is a no-op); = SOURCE dtype for a bool target (the nonzero test must
    # see the source values -- converting 0.5 to bool numerically would give 0).
    # BOOL operands run as their int8 storage view (cutlass.Boolean is 1-BIT; the
    # copy atom rejects it). A bool SOURCE also routes through _to_bool: aten
    # treats ANY nonzero byte as True (a bitcast-produced bool can hold e.g. 2 --
    # see test_non_standard_bool_values), so bool->numeric must produce 0/1 from
    # the nonzero test, not copy the raw byte. A bool TARGET likewise stores the
    # 0/1 test result into its int8 view.
    ret = target
    to_bool = target.dtype is torch.bool or src.dtype is torch.bool
    if tuple(src.shape) != tuple(target.shape):
        # copy_ broadcasts src UP to self; expand here so the kernel's plan (whose
        # shape derives from the inputs) covers the full target. Size-1 dims become
        # stride-0 -> the strided path serves them (vec declines on non-contig).
        src = src.expand(target.shape)
    if src.dtype is torch.bool:
        src = src.view(torch.int8)
    if target.dtype is torch.bool:
        target = target.view(torch.int8)
    if to_bool:
        compute = src.dtype
    elif _needs_int64_detour(src.dtype, target.dtype):
        compute = torch.int64
    else:
        compute = target.dtype
    # compute + to_bool ARE in the key: after the bool->int8 view, f32->bool and
    # f32->int8 share (src, layout, target-int8) but need different fn/compute --
    # without these fields the two would collide on one cached plan (a real bug:
    # the full-matrix sweep caught f32->bool serving f32->int8's identity kernel).
    key = (
        aten_name,
        (src.dtype,),
        ((src.shape, src.stride()),),
        (target.dtype,),
        (),
        compute,
        to_bool,
    )
    K.run(
        ops.get_fn("_to_bool" if to_bool else "_identity"),
        key,
        1,
        1,
        (),
        _L.torch2cute[compute],
        compute,
        [src],
        (target.dtype,),
        out=[target],
    )
    return ret


def _to_copy_cond(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
    **kw,
):
    # Serve only the pure DTYPE cast (or same-dtype clone): any layout/device/pin
    # change is not a pointwise kernel's job -> aten. memory_format must preserve
    # the (contiguous, per _supported) input layout.
    if kw or layout not in (None, torch.strided) or pin_memory:
        return False
    if device is not None and torch.device(device) != x.device:
        return False
    if memory_format not in (None, torch.preserve_format, torch.contiguous_format):
        return False
    dst = x.dtype if dtype is None else dtype
    # The gapped-input rowvec path preserves the gap only in the output we
    # allocate contiguous -> restrict _to_copy to contiguous sources.
    return _conv_serveable(x, dst) and x.is_contiguous()


def _to_copy_impl(x, *, dtype=None, memory_format=None, **kw):
    dst = x.dtype if dtype is None else dtype
    out = torch.empty(x.shape, device=x.device, dtype=dst)
    return _run_conversion("_to_copy", x, out)


def _copy_cond(self, src, non_blocking=False):
    # copy_ casts ANY dtype pair (no can_cast gate -- f32.copy_ into int32 is
    # legal and truncates) and broadcasts src UP to self. Cross-device copies and
    # non-tensor srcs -> aten.
    if not (isinstance(src, torch.Tensor) and src.device == self.device):
        return False
    if not (_conv_serveable(src, self.dtype) and self.is_contiguous()):
        return False
    if cap.is_traced(self):
        return False
    try:
        bshape = torch.broadcast_shapes(self.shape, src.shape)
    except RuntimeError:
        return False
    return tuple(bshape) == tuple(self.shape)


def _copy_impl(self, src, non_blocking=False):
    return _run_conversion("copy_", src, self)


def register_pointwise_overrides() -> None:
    # One PointwiseDef row -> up to len(POINTWISE_VARIANTS) aten overrides (functional /
    # .out / in-place), all sharing the canonical _run_into kernel. A variant aten does
    # not define for this op (e.g. maximum has no in-place) is skipped via
    # variant_aten_name returning None.
    for row in POINTWISE_DEF_TABLE:
        for variant in POINTWISE_VARIANTS:
            if row.skip_out_variant and variant.out_from == "out_kw":
                continue  # aten redispatches a wrapped number here; see the field doc
            aten_name = variant_aten_name(row.aten, variant)
            if aten_name is None:
                continue
            cu.register_op_override(
                "aten",
                aten_name,
                "CUDA",
                cond=_make_cond(row, variant),
                impl=_make_impl(row, variant),
            )
    cu.register_op_override(
        "aten", "_to_copy", "CUDA", cond=_to_copy_cond, impl=_to_copy_impl
    )
    cu.register_op_override("aten", "copy_", "CUDA", cond=_copy_cond, impl=_copy_impl)
