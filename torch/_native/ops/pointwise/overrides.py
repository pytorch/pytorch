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


def _is_wrapped_scalar(t) -> bool:
    # A 0-d CPU tensor standing in for a python number, as produced by the registry's
    # _scalar_arg_coercer for `x * 2.0`. Its dtype is already aten's weak-promotion
    # result, so it needs no dtype fixup -- only a device transfer.
    return isinstance(t, torch.Tensor) and t.dim() == 0 and t.device.type == "cpu"


def _device_ref(ins):
    # The operand whose device the call runs on: the first that is NOT a coerced python
    # number. Returns None if every operand is one (nothing to compute on). aten places
    # the wrapped scalar in slot 0 for the reflected overloads (rsub.Scalar,
    # remainder.Scalar_Tensor, xlogy.Scalar_Self, bitwise_*.Scalar_Tensor), so slot 0 is
    # not a safe device reference.
    return next((t for t in ins if not _is_wrapped_scalar(t)), None)


def _localize_scalars(ins):
    # Move coerced-number operands onto the tensor operands' device. Cheap (a 4/8-byte
    # H2D of a 0-d tensor) and done once per call, before the kernel wraps operands.
    ref = _device_ref(ins)
    if ref is None:
        return ins
    return tuple(t.to(ref.device) if _is_wrapped_scalar(t) else t for t in ins)


def _supported(t, dtypes) -> bool:
    # COW is not gated: inputs export read-only (launch._ro / ReadOnlyTensorWrapper),
    # so a COW input flows through the kernel without materializing.
    #
    # A NEG or CONJ bit must decline. The bit is lazy metadata, not data: the buffer
    # holds unnegated/unconjugated values, so reading it through dlpack would silently
    # compute on the wrong sign. Worse, it cannot be resolved here -- aten materializes
    # a neg/conj view BY CALLING copy_, which our own copy_ override intercepts, so
    # resolving in the cond recurses until the stack blows (a plain torch.sin on a
    # _neg_view() input hit RecursionError). Declining lets aten resolve it normally.
    return (
        isinstance(t, torch.Tensor)
        and t.dtype in dtypes
        and _layout_ok(t)
        and t.numel() > 0
        and not t.is_neg()
        and not t.is_conj()
        and not cap.is_traced(t)
    )


def _scalars(row: PointwiseDef, args, kwargs, compute=None):
    # Resolve the row's named scalar args, which the caller may pass either
    # positionally (after the nin tensors) or by keyword (e.g. torch.add(x, y,
    # alpha=2)). Missing -> the row's aten default (row.scalar_defaults, aligned with
    # row.scalars; unset -> 1, the add/sub-alpha convention). Tuple in row.scalars order.
    #
    # A name in row.optional_defaults is an aten `Scalar?`/`float?` the caller may omit
    # INDEPENDENTLY, and its fill-in is a callable of the RESULT dtype rather than a fixed
    # value (an omitted nan_to_num posinf is that dtype's max, an omitted clamp bound its
    # infinity). `dtype` is None while the cond runs (promotion is not resolved yet); an
    # optional slot then reports None, which the cond only needs for its is-complex check.
    pos = args[row.nin :]
    opt = row.optional_defaults or {}
    out = []
    for i, name in enumerate(row.scalars):
        if name in kwargs and kwargs[name] is not None:
            out.append(kwargs[name])
            continue
        if i < len(pos) and pos[i] is not None:
            out.append(pos[i])
            continue
        if name in opt:
            out.append(opt[name](compute) if compute is not None else None)
            continue
        out.append(row.scalar_defaults[i] if row.scalar_defaults else 1)
    return tuple(out)


def _mode_of(row: PointwiseDef, args, kwargs):
    # The row's mode-kwarg VALUE for this call (gelu's approximate, div's
    # rounding_mode). aten allows it positionally too, after the tensors and any
    # numeric scalars, so fall back to scanning the positional tail for a str/None.
    if row.mode_kwarg in kwargs:
        return kwargs[row.mode_kwarg]
    tail = args[row.nin + len(row.scalars) :]
    return tail[0] if tail else None


def _mode_fn_and_promotion(row: PointwiseDef, args, kwargs, int_compute=False):
    # (fn name, promotion kind) for a mode-selected row, or (None, None) if the mode
    # is one aten accepts but we do not serve -- the cond then declines and aten runs
    # (or raises for a genuinely invalid mode). int_compute picks the mode's integer
    # variant (mode_int_fns) where the integer math differs.
    mode = _mode_of(row, args, kwargs)
    fns = row.mode_fns or {}
    if mode not in fns:
        return None, None
    promo = (row.mode_promotion or {}).get(mode, row.promotion)
    if int_compute:
        fn = (row.mode_int_fns or {}).get(mode, fns[mode])
    else:
        fn = fns[mode]
    return fn, promo


# (promotion kind, input dtypes) -> (compute dtype, functional result dtype). Memoized
# module-wide: promotion is a pure function of these, shape-independent, and BOTH the
# cond (safe-cast gate) and impl (kernel dtypes) need it, so a shared cache avoids the
# ~4us elementwise_dtypes call on the hot path. compute = fp32 opmath dtype; result =
# the functional output dtype (what an alloc/`.out` result would be before any cast).
_promo_cache: dict = {}


def _result_dtypes(row: PointwiseDef, in_dtypes, promotion=None):
    # promotion overrides row.promotion for mode-selected rows (div's rounding_mode
    # changes the rule: None is true division -> INT_TO_FLOAT, floor/trunc -> DEFAULT).
    promotion = row.promotion if promotion is None else promotion
    got = _promo_cache.get((promotion, in_dtypes))
    if got is None:
        probes = [torch.empty(0, dtype=d, device="cuda") for d in in_dtypes]
        compute, result = elementwise_dtypes(*probes, type_promotion_kind=promotion)
        got = (compute, result)
        _promo_cache[(promotion, in_dtypes)] = got
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
        # The DEVICE reference must be a real device operand, not simply operand 0. aten
        # puts the coerced scalar FIRST for the reflected overloads -- rsub.Scalar is
        # `at::sub(wrapped_scalar, self)`, and remainder.Scalar_Tensor / xlogy.Scalar_Self
        # are declared that way -- so anchoring on ins[0] tested a 0-d CPU tensor for
        # CUDA-ness and declined every such call (measured: `1.0 - t` fired nothing).
        ref = _device_ref(ins)
        if ref is None:  # every operand is a coerced scalar -> nothing to compute on
            return False
        if not (cap.device_ok(ref) and cap.on_current_device(ref)):
            return False
        if any(isinstance(s, complex) for s in _scalars(row, args, kwargs)):
            return False
        promotion = None
        if row.mode_kwarg is not None:
            fn_name, promotion = _mode_fn_and_promotion(row, args, kwargs)
            if fn_name is None:
                return False  # mode we do not serve (or invalid) -> aten
        # A 0-d CPU operand is aten's coerced python number (registry's
        # _scalar_arg_coercer wraps x*2.0's 2.0 at the weak-promotion dtype); the impl
        # moves it onto the device. A dim>0 CPU tensor is a genuine cross-device call
        # and still declines.
        if any(t.device != ref.device and not _is_wrapped_scalar(t) for t in ins):
            return False
        if not _layouts_serveable(ins):
            return False  # gapped operand set the rowvec path can't serve -> aten
        try:
            bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        except RuntimeError:
            return False  # incompatible -> aten raises the precise size-mismatch error
        tgt = _out_tensor(variant, row, args, kwargs)
        if variant.out_from != "alloc":
            # 16-byte alignment is required of the TARGET as well as the inputs: the
            # vec/rowvec wraps promise assumed_align=16 and from_dlpack VALIDATES it
            # ("Misaligned Tensor data on mOuts[0]"). _build_plan compiles against a
            # freshly allocated (always aligned) seed output and CACHES the plan, so a
            # later call reusing that plan with a misaligned out=/self view would raise
            # at launch. A view into a base is easily misaligned (base[1:] on fp64 is
            # 8 B in), so gate it here rather than in the kernel's path picker, which
            # never sees the caller's target.
            if not (
                isinstance(tgt, torch.Tensor)
                and tgt.is_contiguous()
                and tgt.device == _device_ref(ins).device
                and K._is_16b_aligned(tgt)
                and not cap.is_traced(tgt)
            ):
                return False
            # _run_into stores each element AT THE TARGET's dtype, so the target dtype
            # must be one the kernel can lower -- can_cast alone is not enough. A real
            # result safe-casts into a COMPLEX out (can_cast(f64, c128) is True), and
            # aten allows it, but we have no complex store: `torch.add(f64, f64,
            # out=c128)` reached _build_plan and raised KeyError('torch.complex128').
            # (_CONV_SRC_DTYPES, not _CONV_DTYPES: bool is a legitimate .out target for
            # the comparison rows, which store a bool result.)
            if tgt.dtype not in _CONV_SRC_DTYPES:
                return False
            # Safe-cast gate: promotion result must fit the target dtype (else aten raises).
            _, result_dtype = _result_dtypes(
                row, tuple(t.dtype for t in ins), promotion
            )
            if not torch.can_cast(result_dtype, tgt.dtype):
                return False
        if variant.shape_rule == "eq_self":
            # In-place: the broadcast result must fit self exactly (no grow).
            return tuple(bshape) == tuple(tgt.shape)
        return True

    return cond


def _make_impl(row: PointwiseDef, variant):
    def _promo(in_dtypes, promotion=None):
        # compute + functional output dtypes for this input-dtype tuple, off the shared
        # _result_dtypes memo. out_dtypes applies the row's escape hatch (e.g. frexp ->
        # [float, int32]); the common case is [result] * nout.
        compute, result = _result_dtypes(row, in_dtypes, promotion)
        out_dtypes = row.out_dtypes(result) if row.out_dtypes else [result] * row.nout
        return _L.torch2cute[compute], compute, tuple(out_dtypes)

    def _run_into(targets, ins, scalars, in_dtypes, fn_override=None, promotion=None):
        # THE canonical kernel (mirrors aten's structured .out kernel): compute row.fn
        # over the broadcast of `ins` and write into `targets`, casting each element to
        # the TARGET tensor's own dtype. Every variant funnels through here -- they
        # differ ONLY in how `targets` is chosen (below). out_dtypes = the targets'
        # dtypes (NOT the promotion result): .out keeps the out= tensor's dtype and
        # in-place keeps self's, both of which aten casts the compute result into.
        ct, compute, _ = _promo(in_dtypes, promotion)
        consts = tuple(ct(s) for s in scalars)
        out_dtypes = tuple(t.dtype for t in targets)
        # Layout + out dtype are baked into the compiled kernel -> in the cache key.
        # Variant is NOT keyed: two variants with the same operand layout and target
        # dtype compile the identical kernel (the target tensor is a runtime arg), so
        # they correctly share one plan.
        # 16-byte alignment must be part of the key, not just of path selection: two
        # calls with identical shape/stride can differ in alignment (base[0:16] vs
        # base[1:17].view(...)), and the vec/rowvec paths the aligned call picks bake
        # assumed_align=16. Without this, the misaligned call reuses the aligned plan
        # and from_dlpack raises "Misaligned Tensor data on mIns[0]" at launch. The
        # TARGETS are keyed too (mOuts[0] in that message): _build_plan compiles against
        # a freshly allocated, always-aligned seed output, so only the caller's .out /
        # in-place tensor can be misaligned, and it arrives as a runtime arg.
        aligned = tuple(K._is_16b_aligned(t) for t in (*ins, *targets))
        key = (
            row.aten,
            in_dtypes,
            tuple((t.shape, t.stride()) for t in ins),
            out_dtypes,
            scalars,
            fn_override,  # mode-selected fn (gelu tanh vs erf, div floor vs trunc)
            aligned,
        )
        # int_fn: the Int-compute variant of the math for ops whose integer semantics
        # differ from float (fmod/remainder/floor_divide -- truncating int division vs
        # cute.math.floor, which rejects Int).
        fn_name = (
            row.int_fn
            if row.int_fn is not None and not compute.is_floating_point
            else (fn_override or row.fn)
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

    def _targets(args, kwargs, ins, in_dtypes, promotion=None):
        # Choose the tensors _run_into writes to, per variant.out_from (DATA):
        #   alloc  (functional): fresh contiguous tensors of the PROMOTION out dtypes,
        #                        broadcast shape.
        #   out_kw (.out): the out= kwarg tensor(s), resized to the broadcast shape to
        #                  match aten (which keeps the out tensor's dtype).
        #   self   (in-place): operand 0 (its shape already == broadcast shape via cond).
        bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        if variant.out_from == "alloc":
            _, _, promo_out = _promo(in_dtypes, promotion)
            return [
                torch.empty(bshape, device=_device_ref(ins).device, dtype=d)
                for d in promo_out
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
        ins = _localize_scalars(args[: row.nin])
        in_dtypes = tuple(t.dtype for t in ins)
        # Mode rows pick their fn (and possibly promotion) from a string kwarg; the
        # cond already verified the mode is one we serve.
        fn_override, promotion = (
            _mode_fn_and_promotion(row, args, kwargs)
            if row.mode_kwarg is not None
            else (None, None)
        )
        # Optional scalars fill in from the RESULT dtype, so resolve promotion first.
        # Result, not compute: aten saturates nan_to_num at the OUTPUT dtype's finite
        # max (fp16 -> 65504), while compute for a half input is fp32, whose max would
        # overflow back to inf on the store.
        compute, result = _result_dtypes(row, in_dtypes, promotion)
        if row.mode_kwarg is not None and not compute.is_floating_point:
            fn_override, promotion = _mode_fn_and_promotion(
                row, args, kwargs, int_compute=True
            )
        scalars = _scalars(row, args, kwargs, result)
        targets = _targets(args, kwargs, ins, in_dtypes, promotion)
        _run_into(targets, ins, scalars, in_dtypes, fn_override, promotion)
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
    # Alignment is in the key for the same reason as in _run_into: identical shape and
    # stride can still differ in 16-byte alignment, and the vec/rowvec wraps this path
    # picks bake assumed_align=16 (a narrow()/slice source is easily 8 B in).
    key = (
        aten_name,
        (src.dtype,),
        ((src.shape, src.stride()),),
        (target.dtype,),
        (),
        compute,
        to_bool,
        (K._is_16b_aligned(src), K._is_16b_aligned(target)),
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
    # The DESTINATION must be 16-byte aligned too: the conversion kernel compiles
    # against a wrap that promises assumed_align=16 and from_dlpack validates it
    # ("Misaligned Tensor data on mOuts[0]"), and unlike _to_copy -- which allocates its
    # own always-aligned output -- copy_ writes into whatever view the caller passed.
    if cap.is_traced(self) or self.is_neg() or self.is_conj():
        return False
    if not K._is_16b_aligned(self):
        return False
    # An EMPTY destination is a no-op for aten but a zero-element grid for us
    # (cudaErrorInvalidConfiguration, or an invalid cute layout when the shape has a 0
    # extent). _conv_serveable only rejects an empty SOURCE, and copy_ broadcasts, so a
    # 1-element src into a (0, 2) dst reaches here with a perfectly valid source.
    if self.numel() == 0:
        return False
    try:
        bshape = torch.broadcast_shapes(self.shape, src.shape)
    except RuntimeError:
        return False
    return tuple(bshape) == tuple(self.shape)


def _copy_impl(self, src, non_blocking=False):
    return _run_conversion("copy_", src, self)


# ---------------------------------------------------------------------------
# fill_: the NULLARY (nin == 0) case. No input tensor at all -- the value is a Scalar and
# the caller's `self` is both the shape/device/layout source and the destination. This is
# the shape the constructors need: aten's full/zeros/ones are all
# CompositeExplicitAugograd wrappers that lower to empty() + fill_(), so overriding fill_
# serves the whole family without a row per constructor.
_FILL_DTYPES = _SUPPORTED + _INT_DTYPES


def _fill_cond(self, value):
    # A python number or 0-d tensor value; a >0-d value tensor is fill_.Tensor's job and
    # aten raises for it, so decline. Same layout/alignment/lazy-metadata gates as the
    # other in-place overrides -- with no input to speak for it, `self` carries them all.
    #
    # Gates are CAPABILITY only: every target we can compute correctly is served. Layout is
    # NOT gated -- a contiguous, aligned target takes the vectorized path and anything else
    # (transposed, strided, unaligned) falls to the strided route, which bakes the real
    # layout and is correct for all of them. numel > 0 is a genuine limit (an empty tensor
    # is a zero-element grid launch, and a 0-extent cute layout is invalid), as is the
    # neg/conj bit (lazy metadata; see _supported).
    if isinstance(value, torch.Tensor) and value.dim() != 0:
        return False
    if isinstance(value, complex):
        return False
    return (
        isinstance(self, torch.Tensor)
        and self.dtype in _FILL_DTYPES
        and self.numel() > 0
        and not self.is_neg()
        and not self.is_conj()
        and not cap.is_traced(self)
        and cap.device_ok(self)
        and cap.on_current_device(self)
    )


def _fill_impl(self, value):
    # compute == the target dtype: there is no promotion to do (aten casts the Scalar to
    # self's dtype), and no input to convert.
    v = value.item() if isinstance(value, torch.Tensor) else value
    # Compute in the OPMATH dtype (fp32 for a half target), not the target dtype: a
    # Float16/BFloat16 scalar cannot cross the tvm-ffi arg boundary ("Unsupported argument
    # type: Float16"), and the store-side cast to self.dtype is what aten does anyway.
    # This mirrors the table path, whose compute is always the fp32 opmath type for halves.
    compute = (
        torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
    )
    # The OUTPUT's layout must be in the key. For an op with inputs, operand shape/stride
    # carries it, but a nullary op has none -- and the plan bakes shape-dependent choices
    # (the ept/vec_bits config, and the strided path's baked layout), so a plan built for
    # (4096,) cannot serve (7,). The scalar VALUE stays out: it is a runtime kernel arg.
    key = (
        "fill_",
        (),
        ((self.shape, self.stride()),),
        (self.dtype,),
        (),
        compute,
        (K._is_16b_aligned(self),),
    )
    # Box the value in the COMPUTE cute dtype, as the table path does via _promo's `ct`.
    # Passing the raw python float instead silently narrows it to the DSL's default width
    # (fp64 1e20 came back as its fp32 neighbour, 1.00000002e20).
    ct = _L.torch2cute[compute]
    K.run(
        ops.get_fn("_fill"),
        key,
        0,  # nin
        1,  # nout
        (ct(v),),
        _L.torch2cute[compute],
        compute,
        [],  # no inputs
        (self.dtype,),
        out=[self],
    )
    return self


# ---------------------------------------------------------------------------
# Range factories: nullary AND index-consuming. Each element's value is a function of its
# FLAT INDEX alone, which is what aten expresses with gpu_kernel_with_index -- so these
# ride the same nin == 0 plumbing as fill_ plus the kernel's `with_index` flag.
#
# We override only the .out forms (arange.start_out / linspace.out), because those are the
# ones carrying an explicit `CUDA:` dispatch; the functional arange/linspace are
# CompositeExplicitAutograd wrappers that size the result and then call straight into
# these, so overriding the .out form serves both. Crucially, all of aten's SIZE logic
# (compute_arange_size's double-precision ceil, the resize + mismatch warning) runs in that
# C++ wrapper before we are reached -- `out` arrives already correctly sized, so we
# reproduce only the value kernel and never re-derive a length.
_RANGE_DTYPES = _SUPPORTED + _INT_DTYPES


def _range_out_serveable(out) -> bool:
    return (
        isinstance(out, torch.Tensor)
        and out.dtype in _RANGE_DTYPES
        and out.numel() > 0
        and not out.is_neg()
        and not out.is_conj()
        and not cap.is_traced(out)
        and cap.device_ok(out)
        and cap.on_current_device(out)
    )


def _range_compute(dtype):
    # aten computes arange in acc_type<scalar_t, true> -- fp32 for a half output, and the
    # dtype itself otherwise. Half consts also cannot cross the tvm-ffi boundary (see
    # _fill_impl), so this doubles as that fix.
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _arange_out_cond(start, end, step=1, *, out=None):
    if any(isinstance(s, complex) for s in (start, end, step)):
        return False
    return _range_out_serveable(out)


def _arange_out_impl(start, end, step=1, *, out=None):
    compute = _range_compute(out.dtype)
    ct = _L.torch2cute[compute]
    key = (
        "arange.start_out",
        (),
        ((out.shape, out.stride()),),
        (out.dtype,),
        (),
        compute,
        (K._is_16b_aligned(out),),
    )
    K.run(
        ops.get_fn("_arange"),
        key,
        0,
        1,
        (ct(start), ct(step)),
        ct,
        compute,
        [],
        (out.dtype,),
        out=[out],
        with_index=True,
    )
    return out


def _linspace_out_cond(start, end, steps, *, out=None):
    if any(isinstance(s, complex) for s in (start, end)):
        return False
    # steps 0/1 are aten's own special cases (empty result / fill_ with start) and never
    # reach a with-index kernel; let the C++ wrapper keep handling them.
    if not isinstance(steps, int) or steps < 2:
        return False
    return _range_out_serveable(out)


def _linspace_out_impl(start, end, steps, *, out=None):
    # COMPUTE IN FP32 for every narrow output (halves and integers alike); the store does
    # the only narrowing. aten instead runs the whole expression in scalar_t, so a HALF
    # output can differ from us by a ULP -- that is aten carrying less precision, not us
    # being wrong, and matching it would mean deliberately rounding intermediates to fp16,
    # exactly the loss the fp32-compute rule exists to prevent. fp32/fp64 are bit-exact.
    #
    # The one conversion that is SEMANTIC rather than precision: aten casts start/end to
    # scalar_t BEFORE deriving the step, which TRUNCATES for an integer output, so
    # linspace(2.5, 3.5, 3, dtype=int32) steps 2 -> 3 and yields [2, 2, 3], not [2, 3, 3].
    # That changes which values exist, so reproduce it.
    if not out.dtype.is_floating_point:
        start, end = int(start), int(end)
    compute = (
        out.dtype if out.dtype in (torch.float32, torch.float64) else torch.float32
    )
    ct = _L.torch2cute[compute]
    step = (float(end) - float(start)) / (steps - 1)
    key = (
        "linspace.out",
        (),
        ((out.shape, out.stride()),),
        (out.dtype,),
        (steps,),
        compute,
        (K._is_16b_aligned(out),),
    )
    K.run(
        ops.get_fn("_linspace"),
        key,
        0,
        1,
        # halfway is compared against the Int64 index, so it must be Int64 too; `last`
        # (steps-1) participates in the float arithmetic, so it takes the compute dtype.
        (ct(start), ct(end), ct(step), _L.Int64(steps // 2), ct(steps - 1)),
        ct,
        compute,
        [],
        (out.dtype,),
        out=[out],
        with_index=True,
    )
    return out


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
    cu.register_op_override(
        "aten", "fill_.Scalar", "CUDA", cond=_fill_cond, impl=_fill_impl
    )
    cu.register_op_override(
        "aten", "arange.start_out", "CUDA", cond=_arange_out_cond, impl=_arange_out_impl
    )
    cu.register_op_override(
        "aten", "linspace.out", "CUDA", cond=_linspace_out_cond, impl=_linspace_out_impl
    )
