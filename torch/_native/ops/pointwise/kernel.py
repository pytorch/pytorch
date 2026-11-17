import math
from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import torch
from torch._prims_common import (
    get_computation_dtype,
    is_non_overlapping_and_dense_or_false,
    make_contiguous_strides_for,
)

from ... import cutedsl_utils as cu
from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.config import (
    POINTWISE_DENSE_MIN as _DENSE_MIN,
    POINTWISE_EPT_SCHEDULE as _EPT_SCHEDULE,
    POINTWISE_REFERENCE_LANES as _B200_DEVICE_LANES,
    POINTWISE_VEC256_MIN as _VEC256_MIN,
)
from ...cutedsl.dtypes import cute2torch, scalar_arg, torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...cutedsl.types import Index, MathFn, MathResult, NumericType, Scalar
from ...utils.device import check_current
from ...utils.tensor import can_use_32bit_indexing, const_data_ptr
from ._types import CacheKey, DTypes, Load, Path, RowGap, Shape
from .layout import (
    row_gap_view as _row_gap_view,
    tma_layout,
    TMA_SMEM_ALIGNMENT,
    TMA_STAGES,
    TMA_TILE,
    TmaLayout,
)
from .signature import Functor, inspect_functor, resolve_type


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


_VEC_BITS = 128
_PLAN: dict[CacheKey, "_Plan"] = {}
_KERNELS: dict[CacheKey, "Function"] = {}


class _PointwiseConfig(NamedTuple):
    block: int = 256
    ept: int = 1
    vec_bits: int = 128


# Storage width and operand count determine work per thread.


def _choose_config(
    numel: int,
    compute_bits: int,
    hw: _hw.HWCaps | None = None,
    nin: int = 1,
    storage_bits: int | None = None,
    bool_output: bool = False,
) -> "_PointwiseConfig":
    fill = 1.0
    if hw is not None:
        fill = hw.sm_count * hw.max_threads_per_sm / _B200_DEVICE_LANES
    if bool_output and compute_bits in (16, 32) and nin > 0:
        if numel < _DENSE_MIN * fill:
            return _PointwiseConfig()
        return _PointwiseConfig(block=128, ept=compute_bits // 16)
    if compute_bits == 32 and nin > 0:
        if numel < _DENSE_MIN * fill:
            return _PointwiseConfig()
        narrow_unary = storage_bits is not None and storage_bits < 32 and nin == 1
        return _PointwiseConfig(
            ept=4 if narrow_unary else 2, vec_bits=256 if narrow_unary else 128
        )
    norm = numel // max(compute_bits // 16, 1)
    ept = next(e for thresh, e in _EPT_SCHEDULE if norm >= thresh * fill)
    wide = nin > 0 and compute_bits >= _VEC256_MIN[0] and numel >= _VEC256_MIN[1] * fill
    return _PointwiseConfig(ept=ept, vec_bits=256 if wide else 128)


def _vec_width(compute_bits: int, vec_bits: int = _VEC_BITS) -> int:
    return max(vec_bits // compute_bits, 1)


@cute.jit
def _block_base(bid: Index, tid: Index, tile: cutlass.Constexpr) -> Index:
    # Power-of-two tiles round a supported count to at most 2**31.
    if const_expr(0 < tile <= 2**31 and tile & (tile - 1) == 0):
        return bid * tile + tid
    return cutlass.Int64(bid) * tile + tid


@dataclass
class _PointwiseOp:
    fn: MathFn
    nin: int
    nout: int
    nconsts: int
    compute: InitVar[NumericType]
    out_types: Sequence[NumericType]
    input_types: Sequence[NumericType] = ()
    host_scalars: tuple[bool, ...] = ()
    result_types: Sequence[NumericType] = ()
    argument_types: Sequence[NumericType] = field(init=False)

    def __post_init__(self, compute: NumericType) -> None:
        self._init_types(compute, start=0)

    def _init_types(self, compute: NumericType, *, start: int) -> None:
        self.input_types = tuple(self.input_types or (compute,) * self.nin)
        annotations, _ = inspect_functor(self.fn)
        self.argument_types = tuple(
            torch2cute[
                resolve_type(a, cute2torch[t], get_computation_dtype(cute2torch[t]))
            ]
            for a, t in zip(annotations[start : start + self.nin], self.input_types)
        )
        self.host_scalars = self.host_scalars or (False,) * self.nin
        self.result_types = tuple(self.result_types or self.out_types)

    @cute.jit
    def _evaluate(self, vals: tuple[Scalar, ...], consts: Any) -> MathResult:
        if const_expr(self.nconsts == 0):
            return self.fn(*vals)
        return self.fn(*vals, *consts)

    @cute.jit
    def _convert(self, k: int, value: Scalar) -> Scalar:
        expected = self.argument_types[k]
        storage = self.input_types[k]
        if const_expr(value.__class__ is not storage):
            value = storage(value)
        return value if const_expr(value.__class__ is expected) else expected(value)

    @cute.jit
    def _cast_result(self, k: int, value: Scalar) -> Scalar:
        result = self.result_types[k]
        output = self.out_types[k]
        if const_expr(value.__class__ is not result):
            value = result(value)
        return value if const_expr(value.__class__ is output) else output(value)

    @cute.jit
    def _compute_store(
        self,
        regs: list[cute.Tensor],
        outputs: list[cute.Tensor],
        index: Any,
        consts: Any,
    ) -> None:
        dsts = [outputs[j][index] for j in range(self.nout)]
        outs = [cute.make_rmem_tensor_like(dst) for dst in dsts]
        for e in cutlass.range_constexpr(cute.size(dsts[0])):
            vals = tuple(
                self._convert(k, regs[k][e]) for k in range(const_expr(self.nin))
            )
            res = self._evaluate(vals, consts)
            if const_expr(self.nout == 1):
                outs[0][e] = self._cast_result(0, res)
            else:
                for j in cutlass.range_constexpr(self.nout):
                    outs[j][e] = self._cast_result(j, res[j])
        for j in cutlass.range_constexpr(self.nout):
            cute.autovec_copy(outs[j], dsts[j])


@dataclass
class _ElementwiseVec(_PointwiseOp):
    block: int = 256
    ept: int = 1
    vector: int = 1
    scalar_inputs: tuple[bool, ...] = ()

    def __post_init__(self, compute: NumericType) -> None:
        super().__post_init__(compute)
        self.scalar_inputs = self.scalar_inputs or (False,) * self.nin

    @cute.jit
    def __call__(
        self, mIns: Any, mOuts: list[cute.Tensor], consts: Any, stream: cuda.CUstream
    ) -> None:
        outs = [cute.group_modes(t, 0, cute.rank(t) - 1) for t in mOuts]
        ins = None
        if const_expr(self.nin > 0):
            ins = [
                t if self.scalar_inputs[k] else cute.group_modes(t, 0, cute.rank(t) - 1)
                for k, t in enumerate(mIns)
            ]
        nvec = cute.size(outs[0].shape[0])
        tile = const_expr(self.block * self.ept)
        self.kernel(ins, outs, consts, nvec).launch(
            grid=[cute.ceil_div(nvec, tile), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mIns: Any, mOuts: list[cute.Tensor], consts: Any, nvec: cutlass.Int32
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        base = _block_base(block, tid, const_expr(self.block * self.ept))
        scalar_regs = []
        for k in cutlass.range_constexpr(self.nin):
            if const_expr(self.scalar_inputs[k]):
                values = cute.make_rmem_tensor((self.vector,), self.argument_types[k])
                values.fill(
                    self._convert(k, mIns[k] if self.host_scalars[k] else mIns[k][0])
                )
                scalar_regs.append(values)
            else:
                scalar_regs.append(None)
        regs = [
            [
                scalar_regs[k]
                if self.scalar_inputs[k]
                else cute.make_rmem_tensor((self.vector,), mIns[k].element_type)
                for k in range(const_expr(self.nin))
            ]
            for _ in range(const_expr(self.ept))
        ]
        for p in cutlass.range_constexpr(self.ept):
            i = base + p * const_expr(self.block)
            if i < nvec:
                for k in cutlass.range_constexpr(self.nin):
                    if const_expr(not self.scalar_inputs[k]):
                        cute.autovec_copy(mIns[k][i, None], regs[p][k])
        for p in cutlass.range_constexpr(self.ept):
            i = base + p * const_expr(self.block)
            if i < nvec:
                self._compute_store(regs[p], mOuts, (i, None), consts)


@dataclass
class _ElementwiseStrided(_PointwiseOp):
    block: int = 256
    ept: int = 1
    with_index: bool = False

    def __post_init__(self, compute: NumericType) -> None:
        self._init_types(compute, start=int(self.with_index))

    @cute.jit
    def __call__(
        self, mIns: Any, mOuts: list[cute.Tensor], consts: Any, stream: cuda.CUstream
    ) -> None:
        n = cute.size(mOuts[0])
        tile = const_expr(self.block * self.ept)
        self.kernel(mIns, mOuts, consts, n).launch(
            grid=[cute.ceil_div(n, tile), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mIns: Any, mOuts: list[cute.Tensor], consts: Any, n: cutlass.Int32
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        block = const_expr(self.block)
        base = _block_base(bidx, tidx, const_expr(block * self.ept))
        for p in cutlass.range_constexpr(self.ept):
            i = base + p * block
            if i < n:
                vals = tuple(
                    self._convert(k, mIns[k] if self.host_scalars[k] else mIns[k][i])
                    for k in range(const_expr(self.nin))
                )
                # Formulas receive Int64 indices; counts and offsets must fit Int32.
                if const_expr(self.with_index):
                    vals = (cutlass.Int64(i), *vals)
                outs = self._evaluate(vals, consts)
                if const_expr(self.nout == 1):
                    mOuts[0][i] = self._cast_result(0, outs)
                else:
                    for j in cutlass.range_constexpr(self.nout):
                        mOuts[j][i] = self._cast_result(j, outs[j])


_VEC_OUT_DTYPES = (
    torch.bool,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.int8,  # Also backs the bool view.
    torch.int16,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)


def _vec_ok(
    inputs: Sequence[torch.Tensor],
    shape: Shape,
    out_dtypes: DTypes,
    V: int,
    out_ref: torch.Tensor | None = None,
) -> bool:
    if any(d not in _VEC_OUT_DTYPES for d in out_dtypes):
        return False
    if out_ref is not None and not _is_16b_aligned(out_ref):
        return False
    numel = math.prod(shape)
    if numel == 0 or numel % V != 0:
        return False
    probes = (
        [t for t in inputs if t.numel() != 1]
        if inputs
        else ([out_ref] if out_ref is not None else [])
    )
    if not probes:
        return False
    if not all(_is_16b_aligned(t) for t in probes):
        return False
    if any(tuple(t.shape) != tuple(shape) for t in probes):
        return False
    if all(t.is_contiguous() for t in probes):
        return out_ref is None or out_ref.is_contiguous()
    return (
        out_ref is not None
        and out_ref.stride() == probes[0].stride()
        and all(
            t.stride() == probes[0].stride()
            and is_non_overlapping_and_dense_or_false(t)
            for t in probes
        )
    )


def _is_16b_aligned(t: torch.Tensor) -> bool:
    return const_data_ptr(t) % 16 == 0


def _view_vec(t: torch.Tensor, V: int) -> torch.Tensor:
    return t.as_strided((t.numel() // V, V), (V, 1))


def _fake_vec(dtype: NumericType, V: int) -> cute.Tensor:
    # Symbolic counts reuse kernels; match operand views in layout and alignment.
    return _L.fake_compact(dtype, (_L.sym(), V), stride_order=(1, 0), align=16)


def _view_rowvec(
    t: torch.Tensor, rows: int, run: int, row_step: int, V: int
) -> torch.Tensor:
    # Group chunks before rows so linear indexing advances along each dense run.
    return t.as_strided((run // V, rows, V), (V, row_step, 1))


def _fake_rowvec(
    t: torch.Tensor, rows: int, run: int, row_step: int, V: int
) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        torch2cute[t.dtype],
        (run // V, rows, V),
        stride=(V, row_step, 1),
        assumed_align=16,
    )


def _strided_view(t: torch.Tensor, shape: Shape, with_index: bool) -> torch.Tensor:
    value = t.expand(shape)
    return value.permute(tuple(reversed(range(value.ndim)))) if with_index else value


def _fake_strided(
    t: torch.Tensor, shape: Shape, with_index: bool = False
) -> cute.Tensor:
    # The cache key includes these concrete shapes and strides.
    v = _strided_view(t, shape, with_index)
    return cute.runtime.make_fake_tensor(
        torch2cute[t.dtype], tuple(v.shape), stride=v.stride()
    )


def _rowvec_ok(
    inputs: Sequence[torch.Tensor],
    shape: Shape,
    out_dtypes: DTypes,
    V: int,
) -> RowGap | None:
    if not inputs:
        return None
    if any(d not in _VEC_OUT_DTYPES for d in out_dtypes):
        return None
    if any(tuple(t.shape) != tuple(shape) for t in inputs):
        return None
    if not all(_is_16b_aligned(t) for t in inputs):
        return None
    view = _row_gap_view(inputs[0])
    if view is None or view[1] % V:
        return None
    return view if all(_row_gap_view(t) == view for t in inputs[1:]) else None


class _Plan(NamedTuple):
    path: Path
    shape: Shape
    V: int
    out_dtypes: DTypes
    fn: "Function"
    rowgap: RowGap | None = None
    tma: TmaLayout | None = None


def tma_eligible(
    inputs: Sequence[torch.Tensor],
    shape: Shape,
    dtype: torch.dtype,
    target: torch.Tensor | None = None,
    *,
    force: bool = False,
    out_strides: Shape | None = None,
) -> TmaLayout | None:
    tensors = [t for t in inputs if t.numel() != 1]
    if not tensors or len(shape) < 1:
        return None
    ref = tensors[0]
    storage = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
    )
    if ref.dtype not in storage or dtype not in (*storage, torch.bool):
        return None
    if any(
        t.device != ref.device
        or t.is_neg()
        or t.is_conj()
        or not can_use_32bit_indexing(t)
        or (t.numel() != 1 and (t.dtype != ref.dtype or tuple(t.shape) != shape))
        for t in inputs
    ):
        return None
    if not ref.is_cuda:
        return None
    hw = _hw.caps(ref.device)
    if hw.cc[0] not in cu.POINTWISE_ARCH_MAJORS:
        return None
    if target is not None and (
        tuple(target.shape) != shape
        or target.dtype != dtype
        or target.device != ref.device
        or target.is_neg()
        or target.is_conj()
        or any(torch._C._is_alias_of(target, t) for t in inputs)
    ):
        return None
    layout = tma_layout(
        tuple((t.dtype, tuple(t.shape), t.stride()) for t in inputs),
        shape,
        dtype,
        target.stride()
        if target is not None
        else out_strides
        if out_strides is not None
        else make_contiguous_strides_for(shape),
    )
    if layout is None:
        return None
    # Small padded-only calls measured slower than ATen; keep them on the fallback.
    if not force and not any(layout.transposed) and math.prod(shape) < 16 * 1024 * 1024:
        return None
    if any(
        orientation is not None and const_data_ptr(t) % (128 if orientation else 16)
        for t, orientation in zip(inputs, layout.transposed)
    ) or (target is not None and const_data_ptr(target) % 16):
        return None
    ntma = sum(t is not None for t in layout.transposed)
    smem = ntma * (TMA_STAGES * TMA_TILE**2 * ref.element_size() + TMA_SMEM_ALIGNMENT)
    smem += TMA_STAGES * 8
    return layout if smem <= hw.smem_per_block_optin else None


def _build_plan(
    fn: MathFn,
    op_name: str,
    nin: int,
    nout: int,
    consts: Sequence[Scalar],
    compute: NumericType,
    inputs: Sequence[torch.Tensor],
    out_dtypes: DTypes,
    block: int | None,
    ept: int | None,
    vec_bits: int | None,
    load: Load | None,
    out_refs: Sequence[torch.Tensor] | None = None,
    with_index: bool = False,
    input_casts: DTypes | None = None,
    result_dtypes: DTypes | None = None,
) -> _Plan:
    ref = (
        next((t for t in inputs if t.is_cuda), None)
        if inputs
        else out_refs[0]
        if out_refs
        else None
    )
    if ref is None:
        raise AssertionError("nullary pointwise plan needs a target tensor")
    shape = (
        tuple(torch.broadcast_shapes(*(t.shape for t in inputs)))
        if inputs
        else tuple(ref.shape)
    )
    if math.prod(shape) >= 2**31 - 1 or any(
        not can_use_32bit_indexing(t) for t in (*inputs, ref, *(out_refs or ()))
    ):
        raise ValueError("pointwise plan requires 32-bit counts and element offsets")
    # Device-scaled measured defaults apply unless the caller supplies tuning values.
    try_tma = load != "direct"
    force_tma = load == "tma"
    row_ept = 1 if ept is None else ept
    hw = _hw.caps(ref.device)
    storage_bits = max((t.element_size() * 8 for t in inputs), default=compute.width)
    cfg = _choose_config(
        math.prod(shape),
        compute.width,
        hw,
        nin,
        storage_bits,
        any(d is torch.bool for d in out_dtypes),
    )
    block = cfg.block if block is None else block
    ept = cfg.ept if ept is None else ept
    vec_bits = cfg.vec_bits if vec_bits is None else vec_bits
    out_types = [torch2cute[torch.int8 if d is torch.bool else d] for d in out_dtypes]
    result_types = tuple(torch2cute[d] for d in (result_dtypes or out_dtypes))
    V = _vec_width(compute.width, vec_bits)
    dev = ref.device
    seed_outs = (
        list(out_refs)
        if out_refs is not None
        else [torch.empty(shape, device=dev, dtype=d) for d in out_dtypes]
    )
    storage_outs = [
        o.view(torch.int8) if o.dtype is torch.bool else o for o in seed_outs
    ]
    rowgap = None
    in_dtypes = tuple(t.dtype for t in inputs)
    common = (
        ref.get_device(),
        hw.cc,
        fn,
        nin,
        nout,
        tuple(type(c) for c in consts),
        compute,
        tuple(out_dtypes),
        in_dtypes,
        with_index,
        input_casts,
        result_dtypes,
    )
    compute_dtype = cute2torch[compute]
    common_dtype = compute_dtype
    if inputs and input_casts is None:
        common_dtype = (
            Functor("_identity")
            .signature(tuple((t.dtype, t.dim() == 0) for t in inputs))
            .storage
        )
        if not common_dtype.is_floating_point and compute_dtype.is_floating_point:
            common_dtype = compute_dtype
    casts = input_casts or tuple(
        common_dtype if t.numel() == 1 and t.is_cuda else compute_dtype for t in inputs
    )
    input_types = tuple(torch2cute[d] for d in casts)
    scalar_inputs = tuple(t.numel() == 1 for t in inputs)
    host_scalars = tuple(t.device.type == "cpu" and t.dim() == 0 for t in inputs)
    common += (input_types, scalar_inputs, host_scalars)
    # Index-aware formulas need the strided path for individual element indices.
    vec1 = not with_index and all(
        _vec_ok(inputs, shape, out_dtypes, V, o) for o in seed_outs
    )
    rg = (
        _rowvec_ok(inputs, shape, out_dtypes, V)
        if not with_index
        and all(o.is_contiguous() and _is_16b_aligned(o) for o in seed_outs)
        else None
    )
    tma = (
        tma_eligible(inputs, shape, out_dtypes[0], seed_outs[0], force=force_tma)
        if try_tma
        and nout == 1
        and not with_index
        and (
            force_tma
            or (not vec1 and any(not t.is_contiguous() for t in (*inputs, *seed_outs)))
        )
        else None
    )
    if tma is not None:
        from .tma import ElementwiseTma

        storage = next(t.dtype for t in inputs if t.numel() != 1)
        out_type = out_types[0]
        op = ElementwiseTma(
            fn,
            len(consts),
            compute,
            out_type,
            torch2cute[storage],
            tma,
            input_types,
            result_types=result_types,
        )
        cin, cout = op.fake_args(tma, tuple(torch2cute[t.dtype] for t in inputs))
        path = "tma"
        kkey = common + (path, tma)
    elif vec1 or rg is not None:
        if rg is None:
            cin = [
                input_types[k](1)
                if host_scalars[k]
                else cute.runtime.make_fake_tensor(
                    torch2cute[t.dtype],
                    (1,),
                    stride=(1,),
                    assumed_align=t.element_size(),
                )
                if scalar_inputs[k]
                else _fake_vec(torch2cute[t.dtype], V)
                for k, t in enumerate(inputs)
            ]
            cout = [_fake_vec(torch2cute[o.dtype], V) for o in storage_outs]
        else:
            # Grouped views default to one vector per thread.
            ept = row_ept
            rows, run, row_step = rg
            cin = [_fake_rowvec(t, rows, run, row_step, V) for t in inputs]
            cout = [_fake_rowvec(o, rows, run, run, V) for o in storage_outs]
            rowgap = rg
        op = _ElementwiseVec(
            fn,
            nin,
            nout,
            len(consts),
            compute,
            out_types,
            input_types,
            host_scalars=host_scalars,
            block=block,
            ept=ept,
            vector=V,
            scalar_inputs=scalar_inputs,
            result_types=result_types,
        )
        path = "vec"
        kkey = common + (path, rg, V, block, ept)
    else:
        op = _ElementwiseStrided(
            fn,
            nin,
            nout,
            len(consts),
            compute,
            out_types,
            input_types,
            block=block,
            ept=ept,
            with_index=with_index,
            host_scalars=host_scalars,
            result_types=result_types,
        )
        cin = [
            input_types[k](1)
            if host_scalars[k]
            else _fake_strided(t, shape, with_index)
            for k, t in enumerate(inputs)
        ]
        cout = [_fake_strided(o, tuple(o.shape), with_index) for o in storage_outs]
        path = "strided"
        lays = tuple((tuple(t.shape), t.stride()) for t in inputs)
        out_lays = tuple((tuple(o.shape), o.stride()) for o in seed_outs)
        kkey = common + ("strided", shape, lays, out_lays, block, ept)
    seed_consts = [c.__class__(1) for c in consts]
    compile_ins = cin if cin else None
    compile_consts = seed_consts if seed_consts else None
    compiled = cached_plan(
        _KERNELS,
        kkey,
        lambda: _L.compile_kernel(op, compile_ins, cout, compile_consts, _L.stream()),
        op=op_name,
    )
    return _Plan(path, shape, V, tuple(out_dtypes), compiled, rowgap, tma)


def run(
    fn: MathFn,
    key: CacheKey,
    nin: int,
    nout: int,
    consts: Sequence[Scalar],
    compute: NumericType,
    inputs: Sequence[torch.Tensor],
    out_dtypes: DTypes,
    block: int | None = None,
    ept: int | None = None,
    vec_bits: int | None = None,
    out: Sequence[torch.Tensor] | None = None,
    with_index: bool = False,
    input_casts: DTypes | None = None,
    result_dtypes: DTypes | None = None,
    *,
    load: Load | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Run matching nin/nout on the current CUDA device, with CPU inputs limited to 0D.
    Callers validate aliases and supplied outputs' broadcast shape, dtype and device."""
    ref = (
        next((t for t in inputs if t.is_cuda), None)
        if inputs
        else out[0]
        if out
        else None
    )
    if ref is None:
        raise AssertionError("nullary pointwise run needs a target tensor")
    device = ref.get_device()
    check_current(device)
    shape = (
        tuple(torch.broadcast_shapes(*(t.shape for t in inputs)))
        if inputs
        else tuple(ref.shape)
    )
    if math.prod(shape) == 0:
        if out is not None:
            if len(out) != nout or any(
                tuple(t.shape) != shape or t.dtype != dtype or t.device != ref.device
                for t, dtype in zip(out, out_dtypes)
            ):
                raise ValueError("empty outputs must match shape, dtype and device")
            return tuple(out)
        return tuple(torch.empty(shape, device=ref.device, dtype=d) for d in out_dtypes)
    layouts = tuple(
        (
            t.device,
            t.dtype,
            tuple(t.shape),
            t.stride(),
            const_data_ptr(t) % 128 if t.is_cuda else 0,
        )
        for t in (*inputs, *(out or ()))
    )
    plan_key = (
        device,
        key,
        fn,
        nin,
        nout,
        tuple(type(c) for c in consts),
        compute,
        tuple(out_dtypes),
        with_index,
        block,
        ept,
        vec_bits,
        load,
        layouts,
        input_casts,
        result_dtypes,
    )
    plan = cached_plan(
        _PLAN,
        plan_key,
        lambda: _build_plan(
            fn,
            f"aten::{key[0]}",
            nin,
            nout,
            consts,
            compute,
            inputs,
            out_dtypes,
            block,
            ept,
            vec_bits,
            load,
            out,
            with_index,
            input_casts,
            result_dtypes,
        ),
    )
    dev = ref.device
    outs = (
        list(out)
        if out is not None
        else [torch.empty(plan.shape, device=dev, dtype=d) for d in plan.out_dtypes]
    )
    storage_outs = [o.view(torch.int8) if o.dtype is torch.bool else o for o in outs]
    if plan.tma is not None:
        cin = [_L.read_only(t.as_strided(*v)) for t, v in zip(inputs, plan.tma.inputs)]
        cout = [storage_outs[0].as_strided(*plan.tma.output)]
    elif plan.rowgap is not None:
        rows, run, row_step = plan.rowgap
        cin = [
            _L.read_only(_view_rowvec(t, rows, run, row_step, plan.V)) for t in inputs
        ]
        cout = [_view_rowvec(o, rows, run, run, plan.V) for o in storage_outs]
    else:
        vector = plan.path == "vec"
        cin = [
            scalar_arg(t.item(), input_casts[k] if input_casts else cute2torch[compute])
            if t.device.type == "cpu" and t.dim() == 0
            else _L.read_only(
                (t.as_strided((1,), (1,)) if t.numel() == 1 else _view_vec(t, plan.V))
                if vector
                else _strided_view(t, plan.shape, with_index)
            )
            for k, t in enumerate(inputs)
        ]
        cout = [
            _view_vec(o, plan.V)
            if vector
            else _strided_view(o, tuple(o.shape), with_index)
            for o in storage_outs
        ]
    # Scalar values are runtime arguments; changing them does not recompile kernels.
    plan.fn(cin or None, cout, list(consts) or None, _L.stream())
    return tuple(outs)
