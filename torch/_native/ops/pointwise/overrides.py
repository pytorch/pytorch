from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import torch
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind,
    is_non_overlapping_and_dense_or_false,
    make_contiguous_strides_for,
)
from torch._tensor_iterator import nullary_op, TensorIterator

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from ...utils.scalar import check_scalar, f32, round_float
from ...utils.tensor import can_use_32bit_indexing, const_data_ptr
from .layout import output_strides, row_gap_view
from .table import (
    _INT_DTYPES,
    POINTWISE_DEF_TABLE,
    POINTWISE_VARIANTS,
    PointwiseDef,
    PointwiseVariant,
    variant_aten_name,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._types import (
        CacheKey,
        Number,
        OperatorResult,
        PromotionOperands,
        TransposeSpec,
    )
    from .signature import Functor, Signature


if TYPE_CHECKING:
    from ...cutedsl import dtypes as dt, launch as _L
    from . import kernel as K, ops, transpose_copy as TC
else:
    # Keep cutlass and kernel imports lazy during `import torch`.
    dt = LazyModule("torch._native.cutedsl.dtypes")
    _L = LazyModule("torch._native.cutedsl.launch")
    K = LazyModule("torch._native.ops.pointwise.kernel")
    ops = LazyModule("torch._native.ops.pointwise.ops")
    TC = LazyModule("torch._native.ops.pointwise.transpose_copy")


_ARCH_MAJORS = cu.POINTWISE_ARCH_MAJORS

_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _layouts_serveable(ins: Sequence[torch.Tensor]) -> bool:
    arrays = [t for t in ins if t.numel() != 1]
    if arrays and all(
        t.shape == arrays[0].shape
        and t.stride() == arrays[0].stride()
        and is_non_overlapping_and_dense_or_false(t)
        for t in arrays
    ):
        return True
    gaps = {row_gap_view(t) for t in ins if not t.is_contiguous()}
    if not gaps:
        return True
    if None in gaps:
        return False
    if len(gaps) > 1:
        return False  # Different pitches need the TMA planner.
    # Contiguous operands have different row geometry from padded operands.
    return all(row_gap_view(t) in gaps for t in ins)


def _is_wrapped_scalar(t: torch.Tensor) -> bool:
    # Registry CPU wrappers retain Python numbers' weak promotion.
    return isinstance(t, torch.Tensor) and t.dim() == 0 and t.device.type == "cpu"


def _device_ref(ins: Sequence[torch.Tensor]) -> torch.Tensor | None:
    return next((t for t in ins if not _is_wrapped_scalar(t)), None)


def _supported(t: object, dtypes: Sequence[torch.dtype]) -> bool:
    # Resolving lazy neg/conj views can redispatch through copy_.
    return (
        isinstance(t, torch.Tensor)
        and t.dtype in dtypes
        and t.numel() > 0
        and can_use_32bit_indexing(t)
        and not t.is_neg()
        and not t.is_conj()
        and not cap.is_traced(t)
    )


def _scalars(
    row: PointwiseDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compute: torch.dtype | None = None,
) -> tuple[Any, ...]:
    # Optional defaults depend on the result dtype; conditions leave them unresolved.
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
    if row.aten == "celu" and not isinstance(out[0], complex):
        alpha = out[0]
        return alpha, 1.0, 1.0 / float(alpha) if alpha != 0 else 0.0
    return tuple(out)


def _mode_of(row: PointwiseDef, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if row.mode_kwarg in kwargs:
        return kwargs[row.mode_kwarg]
    tail = args[row.nin + len(row.scalars) :]
    return tail[0] if tail else None


def _mode_fn_and_promotion(
    row: PointwiseDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Functor | None, PromotionKind | None]:
    mode = _mode_of(row, args, kwargs)
    fns = row.mode_fns or {}
    if mode not in fns:
        return None, None
    promo = (row.mode_promotion or {}).get(mode, row.promotion)
    return fns[mode], promo


def _promo_key(ins: Sequence[torch.Tensor]) -> PromotionOperands:
    r"""Operand dtypes and scalar/tensor promotion tiers."""
    return tuple((t.dtype, t.dim() == 0) for t in ins)


def _signature(
    row: PointwiseDef,
    in_dtypes: PromotionOperands,
    promotion: PromotionKind | None = None,
    fn: Functor | None = None,
) -> Signature:
    promotion = row.promotion if promotion is None else promotion
    return (fn or row.fn).signature(in_dtypes, promotion, row.nout)


def _out_tensor(
    variant: PointwiseVariant,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if variant == "out":
        return kwargs.get("out")
    if variant == "inplace":
        return args[0] if args else None
    return None


def _has_partial_overlap(out: torch.Tensor, inp: torch.Tensor) -> bool:
    # Allow exact dense aliases; defer partial or unresolved overlap to ATen.
    if not torch._C._is_alias_of(out, inp):
        return False
    if (
        out.dtype == inp.dtype
        and out.shape == inp.shape
        and out.stride() == inp.stride()
        and const_data_ptr(out) == const_data_ptr(inp)
        and is_non_overlapping_and_dense_or_false(inp)
    ):
        return False
    if not inp.is_contiguous():
        return True
    out_begin = const_data_ptr(out)
    inp_begin = const_data_ptr(inp)
    out_end = out_begin + out.numel() * out.element_size()
    inp_end = inp_begin + inp.numel() * inp.element_size()
    if out_begin == inp_begin and out_end == inp_end:
        return out.stride() != inp.stride()
    return out_begin < inp_end and inp_begin < out_end


def _make_cond(row: PointwiseDef, variant: PointwiseVariant) -> Callable[..., bool]:
    base = _SUPPORTED if row.dtypes is None else row.dtypes
    ints = row.int_dtypes or (_INT_DTYPES if row.int_via_float else ())
    dtypes = tuple(base) + tuple(ints)

    def cond(*args: Any, **kwargs: Any) -> bool:
        ins = args[: row.nin]
        if len(ins) < row.nin or not all(_supported(t, dtypes) for t in ins):
            return False
        cpu_inputs = [i for i, t in enumerate(ins) if _is_wrapped_scalar(t)]
        if len(cpu_inputs) > 1 or (
            row.cpu_scalar_inputs is not None
            and any(i not in row.cpu_scalar_inputs for i in cpu_inputs)
        ):
            return False
        ref = _device_ref(ins)
        if ref is None:
            return False
        if not (cap.device_ok(ref, _ARCH_MAJORS) and cap.on_current_device(ref)):
            return False
        if (
            any(_is_wrapped_scalar(t) for t in ins)
            and torch.cuda.is_current_stream_capturing()
        ):
            return False
        if row.aten in ("div.Tensor", "div.Tensor_mode") and any(
            _is_wrapped_scalar(t) for t in ins
        ):
            return False
        scalars = _scalars(row, args, kwargs)
        if any(isinstance(s, complex) for s in scalars):
            return False
        if row.aten in ("add.Tensor", "sub.Tensor", "rsub.Tensor"):
            alpha = scalars[0]
            if isinstance(alpha, bool) or (
                isinstance(alpha, float)
                and not any(t.dtype.is_floating_point for t in ins)
            ):
                return False
        if row.aten == "celu" and scalars[0] == 0:
            return False
        if row.aten == "softshrink":
            lambd = scalars[0]
            if not (0 <= lambd <= torch.finfo(ins[0].dtype).max):
                return False
        if row.aten == "clamp":
            if all(s is None for s in scalars):
                return False
            if not ins[0].dtype.is_floating_point and any(
                isinstance(s, float) for s in scalars
            ):
                return False
        if row.aten in ("clamp", "hardtanh", "hardshrink", "softshrink", "threshold"):
            if ins[0].dtype.is_floating_point:
                try:
                    for s in scalars:
                        if s is not None:
                            round_float(float(s), ins[0].dtype)
                except (OverflowError, ValueError):
                    return False
        promotion = None
        fn_name = None
        if row.mode_kwarg is not None:
            fn_name, promotion = _mode_fn_and_promotion(row, args, kwargs)
            if fn_name is None:
                return False
            # Float div floor/trunc via fmod is inexact; defer to ATen's exact kernels.
            if row.mode_int_only and _mode_of(row, args, kwargs) in row.mode_int_only:
                if any(t.dtype.is_floating_point for t in ins):
                    return False
        if any(t.device != ref.device and not _is_wrapped_scalar(t) for t in ins):
            return False
        signature = _signature(row, _promo_key(ins), promotion, fn_name)
        compute, result = signature.compute, signature.results[0]
        try:
            for scalar in scalars:
                if scalar is not None and compute.is_floating_point:
                    round_float(float(scalar), compute)
                elif scalar is not None:
                    limits = torch.iinfo(compute)
                    if not limits.min <= scalar <= limits.max:
                        return False
            for index, value in enumerate(ins):
                if _is_wrapped_scalar(value):
                    scalar = value.item()
                    scalar_dtype = signature.input_dtype(index, True)
                    if value.dtype.is_floating_point and not math.isfinite(scalar):
                        return False
                    if scalar_dtype.is_floating_point:
                        round_float(float(scalar), scalar_dtype)
                    else:
                        limits = torch.iinfo(scalar_dtype)
                        if not limits.min <= scalar <= limits.max:
                            return False
        except (OverflowError, ValueError):
            return False
        basic_layout = _layouts_serveable(ins)
        try:
            bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        except RuntimeError:
            return False
        if math.prod(bshape) >= 2**31 - 1:
            return False
        tgt = _out_tensor(variant, args, kwargs)
        if row.precondition is not None and not row.precondition(ins, tgt):
            return False
        if variant != "functional":
            if not (
                isinstance(tgt, torch.Tensor)
                and is_non_overlapping_and_dense_or_false(tgt)
                and tgt.device == ref.device
                and const_data_ptr(tgt) % 16 == 0
                and not cap.is_traced(tgt)
            ):
                return False
            if any(_has_partial_overlap(tgt, t) for t in ins):
                return False
            if tuple(tgt.shape) != tuple(bshape):
                return False
            if tgt.dtype not in _CONV_SRC_DTYPES:
                return False
            if not torch.can_cast(result, tgt.dtype):
                return False
        if not basic_layout:
            dtype = tgt.dtype if isinstance(tgt, torch.Tensor) else result
            return (
                row.nout == 1
                and K.tma_eligible(
                    ins,
                    tuple(bshape),
                    dtype,
                    tgt,
                    out_strides=output_strides(
                        tuple(
                            (t.dtype, tuple(t.shape), t.stride())
                            for t in (ins[::-1] if row.aten == "rsub.Tensor" else ins)
                        ),
                        dtype,
                    )
                    if variant == "functional"
                    else None,
                )
                is not None
            )
        return True

    return cond


def _make_impl(
    row: PointwiseDef, variant: PointwiseVariant
) -> Callable[..., OperatorResult]:
    def _run_into(
        targets: Sequence[torch.Tensor],
        ins: Sequence[torch.Tensor],
        scalars: Sequence[Any],
        in_dtypes: PromotionOperands,
        signature: Signature,
        fn_override: Functor | None = None,
    ) -> None:
        compute = signature.compute
        ct = dt.torch2cute[compute]
        casts = tuple(
            signature.input_dtype(k, _is_wrapped_scalar(t)) for k, t in enumerate(ins)
        )
        if row.aten in ("clamp", "hardtanh", "hardshrink", "softshrink", "threshold"):
            if ins[0].dtype.is_floating_point:
                scalars = tuple(round_float(s, ins[0].dtype) for s in scalars)
        consts = tuple(dt.scalar_arg(s, compute) for s in scalars)
        out_dtypes = tuple(t.dtype for t in targets)
        aligned = tuple(const_data_ptr(t) % 16 == 0 for t in (*ins, *targets))
        key = (
            row.aten,
            in_dtypes,
            tuple((t.shape, t.stride()) for t in ins),
            out_dtypes,
            tuple(type(c) for c in consts),
            fn_override,
            aligned,
        )
        fn_name = fn_override or row.fn
        K.run(
            ops.get_fn(fn_name),
            key,
            row.nin,
            row.nout,
            consts,
            ct,
            ins,
            out_dtypes,
            out=targets,
            input_casts=casts,
            result_dtypes=signature.results,
        )

    def _targets(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        ins: Sequence[torch.Tensor],
        signature: Signature,
    ) -> list[torch.Tensor]:
        bshape = torch.broadcast_shapes(*(t.shape for t in ins))
        ref = _device_ref(ins)
        if ref is None:
            raise AssertionError("pointwise impl reached with no device operand")
        if variant == "functional":
            operands = tuple(
                (t.dtype, tuple(t.shape), t.stride())
                for t in (ins[::-1] if row.aten == "rsub.Tensor" else ins)
            )
            return [
                torch.empty_strided(
                    bshape, output_strides(operands, d), device=ref.device, dtype=d
                )
                for d in signature.results
            ]
        if variant == "out":
            out = kwargs["out"]
            return list(out) if isinstance(out, (tuple, list)) else [out]
        return [args[0]]

    def impl(*args: Any, **kwargs: Any) -> OperatorResult:
        ins = args[: row.nin]
        in_dtypes = _promo_key(ins)
        fn_override, promotion = (
            _mode_fn_and_promotion(row, args, kwargs)
            if row.mode_kwarg is not None
            else (None, None)
        )
        signature = _signature(row, in_dtypes, promotion, fn_override)
        scalars = _scalars(row, args, kwargs, signature.results[0])
        targets = _targets(args, kwargs, ins, signature)
        _run_into(targets, ins, scalars, in_dtypes, signature, fn_override)
        return targets[0] if row.nout == 1 else tuple(targets)

    return impl


# Complex and float8 conversions use ATen.
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


def _conv_serveable(src: torch.Tensor, dst_dtype: torch.dtype) -> bool:
    # FP64-to-uint16 edges require ATen's conversion.
    if src.dtype is torch.float64 and dst_dtype is torch.uint16:
        return False
    return (
        _supported(src, _CONV_SRC_DTYPES)
        and (
            is_non_overlapping_and_dense_or_false(src) or row_gap_view(src) is not None
        )
        and dst_dtype in _CONV_DTYPES + (torch.bool,)
        and not (dst_dtype is torch.bool and src.dtype is torch.bool)
        and cap.device_ok(src, _ARCH_MAJORS)
        and cap.on_current_device(src)
    )


def _needs_int64_detour(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> bool:
    # Match ATen's narrowing and signed-to-wider-unsigned conversions through Int64.
    if dst_dtype.is_floating_point or dst_dtype is torch.bool:
        return False
    if src_dtype.is_floating_point:
        return dst_dtype in (torch.uint8, torch.int8, torch.int16)
    if dst_dtype.is_signed or not src_dtype.is_signed:
        return False
    return torch.iinfo(dst_dtype).bits > torch.iinfo(src_dtype).bits


def _run_conversion(
    aten_name: str, src: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # Bool storage uses Int8 views and stores normalized zero/one bytes.
    ret = target
    to_bool = target.dtype is torch.bool or src.dtype is torch.bool
    if tuple(src.shape) != tuple(target.shape):
        # Expand copy sources to the target shape; zero strides select generic indexing.
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
    key = (
        aten_name,
        (src.dtype,),
        ((src.shape, src.stride()),),
        (target.dtype,),
        (),
        compute,
        to_bool,
        ((const_data_ptr(src) % 16 == 0), (const_data_ptr(target) % 16 == 0)),
    )
    K.run(
        ops.get_fn("_to_bool" if to_bool else "_identity"),
        key,
        1,
        1,
        (),
        dt.torch2cute[compute],
        [src],
        (target.dtype,),
        out=[target],
    )
    return ret


def _to_copy_cond(
    x: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
    **kw: Any,
) -> bool:
    # Layout, device and pinning changes use ATen.
    if kw or layout not in (None, torch.strided) or pin_memory:
        return False
    if device is not None and torch.device(device) != x.device:
        return False
    if memory_format not in (None, torch.preserve_format, torch.contiguous_format):
        return False
    dst = x.dtype if dtype is None else dtype
    # Rowvec preserves input gaps; _to_copy allocates contiguous output.
    return _conv_serveable(x, dst) and x.is_contiguous()


def _to_copy_impl(
    x: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format | None = None,
    **kw: Any,
) -> torch.Tensor:
    dst = x.dtype if dtype is None else dtype
    out = (
        torch.empty(x.shape, device=x.device, dtype=dst)
        if memory_format is torch.contiguous_format
        else torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dst)
    )
    return _run_conversion("_to_copy", x, out)


def _transpose_source_serveable(
    src: torch.Tensor, quantized: bool, *, check_dense: bool = True
) -> bool:
    if not (
        src.layout is torch.strided
        and src.is_quantized == quantized
        and src.ndim >= 2
        and not src.is_contiguous()
        and (not check_dense or is_non_overlapping_and_dense_or_false(src))
        and src.numel() > 0
        and not src.is_neg()
        and not src.is_conj()
        and not src._is_zerotensor()
        and not cap.is_traced(src)
        and cap.device_ok(src, _ARCH_MAJORS)
        and cap.on_current_device(src)
    ):
        return False
    if quantized and (
        src.dtype
        not in (
            torch.qint8,
            torch.quint8,
            torch.qint32,
            torch.quint4x2,
            torch.quint2x4,
        )
        or src.qscheme() not in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    ):
        return False
    if src.dtype in (torch.quint4x2, torch.quint2x4):
        pack = 2 if src.dtype is torch.quint4x2 else 4
        if src.storage_offset() % pack:
            return False
    return True


def _transpose_source_spec(
    src: torch.Tensor, *, quantized: bool = False
) -> TransposeSpec | None:
    if not _transpose_source_serveable(src, quantized):
        return None
    itemsize = src.element_size()
    lanes = max(1, itemsize // 8)
    if itemsize not in (1, 2, 4, 8, 16) or src.numel() * lanes > 2**31 - 1:
        return None

    shape = tuple(src.shape)
    for dim0 in range(src.ndim):
        for dim1 in range(dim0 + 1, src.ndim):
            base_shape = list(shape)
            base_shape[dim0], base_shape[dim1] = (
                base_shape[dim1],
                base_shape[dim0],
            )
            expected = list(make_contiguous_strides_for(base_shape))
            expected[dim0], expected[dim1] = expected[dim1], expected[dim0]
            if tuple(expected) != src.stride():
                continue
            outer = math.prod(shape[:dim0])
            b = shape[dim0]
            middle = math.prod(shape[dim0 + 1 : dim1])
            a = shape[dim1]
            inner = math.prod(shape[dim1 + 1 :])
            tile = 64 if itemsize <= 4 else 32
            packed = src.dtype in (torch.quint4x2, torch.quint2x4)
            if (
                inner == 1
                and not packed
                and (
                    outer * middle > 65535
                    or (a + tile - 1) // tile > 65535
                    or (b + tile - 1) // tile > 65535
                )
            ):
                return None
            return outer, b, middle, a, inner, itemsize
    return None


_TRANSPOSE_SPECS: dict[CacheKey, TransposeSpec | None] = {}


def _transpose_iterator_spec(
    out: torch.Tensor, src: torch.Tensor
) -> TransposeSpec | None:
    key = (src.dtype, tuple(src.shape), src.stride(), out.stride())
    if key not in _TRANSPOSE_SPECS:
        _TRANSPOSE_SPECS[key] = _build_transpose_iterator_spec(out, src)
    return _TRANSPOSE_SPECS[key]


def _build_transpose_iterator_spec(
    out: torch.Tensor, src: torch.Tensor
) -> TransposeSpec | None:
    try:
        iterator = TensorIterator(
            outputs=[out],
            const_inputs=[src],
            resize_outputs=False,
            check_all_same_dtype=False,
            check_all_same_device=False,
        )
    except RuntimeError:
        return None
    if iterator.ndim < 2:
        return None

    # Fastest-first groups are (inner, a, middle, b, outer); source swaps a and b.
    shape = iterator.shape
    out_strides = iterator.strides(0)
    src_strides = iterator.strides(1)
    stride = src.element_size()
    for dim in range(iterator.ndim):
        if out_strides[dim] != stride:
            return None
        stride *= shape[dim]

    order = sorted(range(iterator.ndim), key=src_strides.__getitem__)
    stride = src.element_size()
    for dim in order:
        if src_strides[dim] != stride:
            return None
        stride *= shape[dim]
    moved = [dim for dim, source_dim in enumerate(order) if dim != source_dim]
    if len(moved) != 2:
        return None
    dim0, dim1 = moved
    if order[dim0] != dim1 or order[dim1] != dim0:
        return None

    inner = math.prod(shape[:dim0])
    a = shape[dim0]
    middle = math.prod(shape[dim0 + 1 : dim1])
    b = shape[dim1]
    outer = math.prod(shape[dim1 + 1 :])
    return outer, b, middle, a, inner, src.element_size()


def _transpose_copy_spec(
    out: torch.Tensor, src: torch.Tensor, *, quantized: bool = False
) -> TransposeSpec | None:
    if not _transpose_source_serveable(src, quantized, check_dense=False) or not (
        out.layout is torch.strided
        and out.is_quantized == quantized
        and out.ndim == src.ndim
        and out.shape == src.shape
        and out.dtype == src.dtype
        and out.device == src.device
        and out.is_contiguous()
        and not out.is_neg()
        and not out.is_conj()
        and not cap.is_traced(out)
        and not torch._C._is_alias_of(out, src)
    ):
        return None
    if quantized and (
        out.qscheme() != src.qscheme()
        or out.q_scale() != src.q_scale()
        or out.q_zero_point() != src.q_zero_point()
    ):
        return None
    if out.dtype in (torch.quint4x2, torch.quint2x4):
        pack = 2 if out.dtype is torch.quint4x2 else 4
        if out.storage_offset() % pack:
            return None
    if src.ndim == 2 and src.stride() == (1, src.shape[0]):
        spec = (1, src.shape[0], 1, src.shape[1], 1, src.element_size())
    else:
        spec = _transpose_iterator_spec(out, src)
        if spec is None:
            return None
    outer, b, middle, a, inner, itemsize = spec
    lanes = max(1, itemsize // 8)
    if itemsize not in (1, 2, 4, 8, 16) or src.numel() * lanes > 2**31 - 1:
        return None
    tile = 64 if itemsize <= 4 else 32
    packed = src.dtype in (torch.quint4x2, torch.quint2x4)
    if (
        inner == 1
        and not packed
        and (
            outer * middle > 65535
            or (a + tile - 1) // tile > 65535
            or (b + tile - 1) // tile > 65535
        )
    ):
        return None
    return spec


def _copy_cond(
    self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False
) -> bool:
    # copy_ allows dtype casts and source broadcasting; cross-device copies use ATen.
    if not (
        isinstance(src, torch.Tensor)
        and src.device == self.device
        and can_use_32bit_indexing(self)
    ):
        return False
    if _transpose_copy_spec(self, src) is not None:
        return True
    if not (_conv_serveable(src, self.dtype) and self.is_contiguous()):
        return False
    if cap.is_traced(self) or self.is_neg() or self.is_conj():
        return False
    if const_data_ptr(self) % 16 != 0:
        return False
    if _has_partial_overlap(self, src):
        return False
    if self.numel() == 0:
        return False
    try:
        bshape = torch.broadcast_shapes(self.shape, src.shape)
    except RuntimeError:
        return False
    return tuple(bshape) == tuple(self.shape)


def _copy_impl(
    self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    spec = _transpose_copy_spec(self, src)
    if spec is not None:
        return TC.run(src, self, spec)
    return _run_conversion("copy_", src, self)


def _quantized_copy_cond(
    self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False
) -> bool:
    return (
        isinstance(src, torch.Tensor)
        and src.device == self.device
        and _transpose_copy_spec(self, src, quantized=True) is not None
    )


def _quantized_copy_impl(
    self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    spec = _transpose_copy_spec(self, src, quantized=True)
    if spec is None:
        raise AssertionError("quantized transpose copy no longer matches")
    return TC.run(src, self, spec)


def _quantized_clone_cond(
    self: torch.Tensor, *, memory_format: torch.memory_format | None = None
) -> bool:
    return (
        memory_format is torch.contiguous_format
        and _transpose_source_spec(self, quantized=True) is not None
    )


def _quantized_clone_impl(
    self: torch.Tensor, *, memory_format: torch.memory_format | None = None
) -> torch.Tensor:
    out = torch.empty_like(self, memory_format=torch.contiguous_format)
    spec = _transpose_copy_spec(out, self, quantized=True)
    if spec is None:
        raise AssertionError("quantized transpose clone no longer matches")
    return TC.run(self, out, spec)


_FILL_DTYPES = _SUPPORTED + _INT_DTYPES


def _fill_cond(self: torch.Tensor, value: Number) -> bool:
    if isinstance(value, complex):
        return False
    valid = (
        isinstance(self, torch.Tensor)
        and self.dtype in _FILL_DTYPES
        and self.numel() > 0
        and can_use_32bit_indexing(self)
        and not self.is_neg()
        and not self.is_conj()
        and not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and cap.on_current_device(self)
    )
    if not valid:
        return False
    try:
        check_scalar(value, self.dtype)
    except (OverflowError, ValueError):
        return False
    return True


def _fill_impl(self: torch.Tensor, value: Number) -> torch.Tensor:
    # Narrow floating fills compute in FP32.
    compute = (
        torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
    )
    key = (
        "fill_",
        (),
        ((self.shape, self.stride()),),
        (self.dtype,),
        (),
        compute,
        ((const_data_ptr(self) % 16 == 0),),
    )
    # Box at compute precision to preserve FP64 constants.
    K.run(
        ops.get_fn("_fill"),
        key,
        0,
        1,
        (dt.scalar_arg(value, compute),),
        dt.torch2cute[compute],
        [],
        (self.dtype,),
        out=[self],
    )
    return self


def _zero_cond(self: torch.Tensor) -> bool:
    return _fill_cond(self, 0)


def _zero_impl(self: torch.Tensor) -> torch.Tensor:
    return _fill_impl(self, 0)


_RANGE_DTYPES = _SUPPORTED + _INT_DTYPES


def _range_out_serveable(out: torch.Tensor | None) -> bool:
    if not (
        isinstance(out, torch.Tensor)
        and out.dtype in _RANGE_DTYPES
        and out.numel() > 0
        and can_use_32bit_indexing(out)
        and not out.is_neg()
        and not out.is_conj()
        and not cap.is_traced(out)
        and cap.device_ok(out, _ARCH_MAJORS)
        and cap.on_current_device(out)
    ):
        return False
    try:
        nullary_op(out)
    except RuntimeError:
        return False
    return True


def _range_compute(dtype: torch.dtype) -> torch.dtype:
    if dtype in _INT_DTYPES:
        return torch.int64
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _arange_out_cond(
    start: Number, end: Number, step: Number = 1, *, out: torch.Tensor | None = None
) -> bool:
    if (
        isinstance(start, complex)
        or isinstance(end, complex)
        or isinstance(step, complex)
    ):
        return False
    if out is None or not _range_out_serveable(out):
        return False
    if not all(math.isfinite(s) for s in (start, end, step)) or step == 0:
        return False
    try:
        compute = _range_compute(out.dtype)
        check_scalar(start, compute)
        check_scalar(step, compute)
    except (OverflowError, ValueError):
        return False
    if (step > 0 and end < start) or (step < 0 and end > start):
        return False
    if out.dtype is torch.int64 and all(type(s) is int for s in (start, end, step)):
        try:
            # Preserve native fallback when its Int64 count intermediates overflow.
            distance = end - start
            numerator = distance + step
            sign = 1 if step > 0 else -1
            for value in (end, distance, numerator, numerator - sign):
                check_scalar(value, compute)
        except (OverflowError, ValueError):
            return False
        count = -((start - end) // step)
    else:
        ratio = (float(end) - float(start)) / float(step)
        if not math.isfinite(ratio):
            return False
        count = math.ceil(ratio)
    return out.numel() == count


def _arange_out_impl(
    start: Number, end: Number, step: Number = 1, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    # Optional out mirrors the schema; admission requires a tensor.
    if out is None:
        raise AssertionError("arange.start_out requires an out tensor")
    compute = _range_compute(out.dtype)
    ct = dt.torch2cute[compute]
    key = (
        "arange.start_out",
        (),
        ((out.shape, out.stride()),),
        (out.dtype,),
        (),
        compute,
        ((const_data_ptr(out) % 16 == 0),),
    )
    K.run(
        ops.get_fn("_arange"),
        key,
        0,
        1,
        (dt.scalar_arg(start, compute), dt.scalar_arg(step, compute)),
        ct,
        [],
        (out.dtype,),
        out=[out],
        with_index=True,
    )
    return out


def _linspace_out_cond(
    start: Number, end: Number, steps: int, *, out: torch.Tensor | None = None
) -> bool:
    if isinstance(start, complex) or isinstance(end, complex):
        return False
    # ATen handles steps 0/1 as empty/fill special cases.
    if not isinstance(steps, int) or steps < 2:
        return False
    if out is None or not _range_out_serveable(out) or out.numel() != steps:
        return False
    if out.dtype.is_floating_point:
        try:
            round_float(float(start), out.dtype)
            round_float(float(end), out.dtype)
        except (OverflowError, ValueError):
            return False
    else:
        limits = torch.iinfo(out.dtype)
        if not all(limits.min <= s <= limits.max for s in (start, end)):
            return False
    return True


def _linspace_out_impl(
    start: bool | int | float,
    end: bool | int | float,
    steps: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # Integer endpoints truncate as in ATen; narrow floats compute in FP32.
    if out is None:
        raise AssertionError("linspace.out requires an out tensor")
    if not out.dtype.is_floating_point:
        start, end = int(start), int(end)
    compute = (
        out.dtype if out.dtype in (torch.float32, torch.float64) else torch.float32
    )
    ct = dt.torch2cute[compute]
    storage = out.dtype if out.dtype.is_floating_point else torch.float32
    start, end = round_float(start, storage), round_float(end, storage)
    step = f32(end - start) if compute is torch.float32 else end - start
    denominator = round_float(float(steps - 1), compute)
    step = round_float(step / denominator, compute)
    key = (
        "linspace.out",
        (),
        ((out.shape, out.stride()),),
        (out.dtype,),
        (steps,),
        compute,
        ((const_data_ptr(out) % 16 == 0),),
    )
    K.run(
        ops.get_fn("_linspace"),
        key,
        0,
        1,
        # Keep index distances integral until subtraction to avoid float rounding.
        (ct(start), ct(end), ct(step), dt.Int64(steps // 2), dt.Int64(steps - 1)),
        ct,
        [],
        (out.dtype,),
        out=[out],
        with_index=True,
    )
    return out


def register_pointwise_overrides() -> None:
    for row in POINTWISE_DEF_TABLE:
        for variant in POINTWISE_VARIANTS:
            if row.skip_out_variant and variant == "out":
                continue
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
        "aten",
        "copy_",
        "QuantizedCUDA",
        cond=_quantized_copy_cond,
        impl=_quantized_copy_impl,
    )
    cu.register_op_override(
        "aten",
        "clone",
        "QuantizedCUDA",
        cond=_quantized_clone_cond,
        impl=_quantized_clone_impl,
    )
    cu.register_op_override(
        "aten", "fill_.Scalar", "CUDA", cond=_fill_cond, impl=_fill_impl
    )
    cu.register_op_override("aten", "zero_", "CUDA", cond=_zero_cond, impl=_zero_impl)
    cu.register_op_override(
        "aten", "arange.start_out", "CUDA", cond=_arange_out_cond, impl=_arange_out_impl
    )
    cu.register_op_override(
        "aten", "linspace.out", "CUDA", cond=_linspace_out_cond, impl=_linspace_out_impl
    )
