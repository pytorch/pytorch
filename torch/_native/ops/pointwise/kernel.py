import math
from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import torch
from torch._prims_common import get_computation_dtype

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, scalar_arg, torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...cutedsl.types import Index, MathFn, MathResult, NumericType, Scalar
from ...utils.device import check_current
from ...utils.tensor import can_use_32bit_indexing, const_data_ptr
from ._types import CacheKey, DTypes, Path, Shape
from .layout import iteration_layout, IterationLayout
from .signature import Functor, inspect_functor, resolve_type


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


_PLAN: dict[CacheKey, "_Plan"] = {}
_KERNELS: dict[CacheKey, "Function"] = {}
_ITERATION_DIMS = 32


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


@dataclass
class _ElementwiseStrided(_PointwiseOp):
    block: int = 256
    elements_per_thread: int = 1
    with_index: bool = False

    def __post_init__(self, compute: NumericType) -> None:
        self._init_types(compute, start=int(self.with_index))

    @cute.jit
    def __call__(
        self,
        mIns: Any,
        mOuts: list[cute.Tensor],
        consts: Any,
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        n = cute.size(mOuts[0])
        tile = const_expr(self.block * self.elements_per_thread)
        self.kernel(mIns, mOuts, consts, n, rank).launch(
            grid=[cute.ceil_div(n, tile), 1, 1],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIns: Any,
        mOuts: list[cute.Tensor],
        consts: Any,
        n: cutlass.Int32,
        rank: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        block = const_expr(self.block)
        base = _block_base(bidx, tidx, const_expr(block * self.elements_per_thread))
        for p in cutlass.range_constexpr(self.elements_per_thread):
            i = base + p * block
            if i < n:
                rest = i
                offsets = [cutlass.Int64(0) for _ in range(self.nin + self.nout)]
                for d in cutlass.range_constexpr(_ITERATION_DIMS):
                    if d < rank:
                        coord = rest
                        if d + 1 < rank:
                            coord = rest % mOuts[0].shape[d]
                            rest = rest // mOuts[0].shape[d]
                        for k in cutlass.range_constexpr(self.nin):
                            if const_expr(not self.host_scalars[k]):
                                offsets[k] += cutlass.Int64(coord) * mIns[k].stride[d]
                        for k in cutlass.range_constexpr(self.nout):
                            offsets[self.nin + k] += (
                                cutlass.Int64(coord) * mOuts[k].stride[d]
                            )
                values = []
                for k in cutlass.range_constexpr(self.nin):
                    value = mIns[k]
                    if const_expr(not self.host_scalars[k]):
                        view = cute.make_tensor(
                            value.iterator + offsets[k], cute.make_layout(1)
                        )
                        value = view[0]
                    values.append(self._convert(k, value))
                vals = tuple(values)
                # Indexed formulas use the original linear order preserved by TI.
                if const_expr(self.with_index):
                    vals = (cutlass.Int64(i), *vals)
                outs = self._evaluate(vals, consts)
                for j in cutlass.range_constexpr(self.nout):
                    target = cute.make_tensor(
                        mOuts[j].iterator + offsets[self.nin + j], cute.make_layout(1)
                    )
                    result = outs if const_expr(self.nout == 1) else outs[j]
                    target[0] = self._cast_result(j, result)


def _fake_iteration(dtype: NumericType) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        tuple(cute.sym_int32() for _ in range(_ITERATION_DIMS)),
        stride=tuple(cute.sym_int64() for _ in range(_ITERATION_DIMS)),
        assumed_align=max(dtype.width // 8, 1),
    )


def _iteration_view(
    tensor: torch.Tensor, layout: IterationLayout, index: int
) -> torch.Tensor:
    padding = _ITERATION_DIMS - len(layout.shape)
    return tensor.as_strided(
        layout.shape + (1,) * padding, layout.strides[index] + (0,) * padding
    )


class _Plan(NamedTuple):
    path: Path
    shape: Shape
    out_dtypes: DTypes
    fn: "Function"
    iteration: IterationLayout


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
    work_per_thread: int | None,
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
    block = 256 if block is None else block
    work_per_thread = 1 if work_per_thread is None else work_per_thread
    hw = _hw.caps(ref.device)
    out_types = [torch2cute[torch.int8 if d is torch.bool else d] for d in out_dtypes]
    result_types = tuple(torch2cute[d] for d in (result_dtypes or out_dtypes))
    dev = ref.device
    seed_outs = (
        list(out_refs)
        if out_refs is not None
        else [torch.empty(shape, device=dev, dtype=d) for d in out_dtypes]
    )
    storage_outs = [
        o.view(torch.int8) if o.dtype is torch.bool else o for o in seed_outs
    ]
    geometry = iteration_layout(list(inputs), seed_outs, with_index=with_index)
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
    host_scalars = tuple(t.device.type == "cpu" and t.dim() == 0 for t in inputs)
    casts = input_casts or tuple(
        compute_dtype if host else common_dtype for host in host_scalars
    )
    input_types = tuple(torch2cute[d] for d in casts)
    generic_key = common + (input_types, host_scalars)
    iteration = geometry
    if len(iteration.shape) > _ITERATION_DIMS:
        raise ValueError("pointwise iterator exceeds 32 coalesced dimensions")
    op = _ElementwiseStrided(
        fn,
        nin,
        nout,
        len(consts),
        compute,
        out_types,
        input_types,
        block=block,
        elements_per_thread=work_per_thread,
        with_index=with_index,
        host_scalars=host_scalars,
        result_types=result_types,
    )
    cin = [
        input_types[k](1) if host_scalars[k] else _fake_iteration(torch2cute[t.dtype])
        for k, t in enumerate(inputs)
    ]
    cout = [_fake_iteration(torch2cute[o.dtype]) for o in storage_outs]
    path = "strided"
    kkey = generic_key + ("strided", block, work_per_thread)
    if input_casts is None:
        annotations, _ = inspect_functor(fn)
        start = int(with_index)
        op.argument_types = tuple(
            torch2cute[resolve_type(a, common_dtype, compute_dtype)]
            for a in annotations[start : start + nin]
        )
    seed_consts = [c.__class__(1) for c in consts]
    compile_ins = cin if cin else None
    compile_consts = seed_consts if seed_consts else None
    args = (compile_ins, cout, compile_consts, cutlass.Int32(1))
    compiled = cached_plan(
        _KERNELS,
        kkey,
        lambda: _L.compile_kernel(op, *args, _L.stream()),
        op=op_name,
    )
    return _Plan(path, shape, tuple(out_dtypes), compiled, iteration)


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
    work_per_thread: int | None = None,
    out: Sequence[torch.Tensor] | None = None,
    with_index: bool = False,
    input_casts: DTypes | None = None,
    result_dtypes: DTypes | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Run matching nin/nout on the current CUDA device, with CPU inputs limited to 0D.
    Default device loads round through the common dtype before ``compute`` math;
    explicit ``input_casts`` bind functor types. Callers validate aliases and outputs."""
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
        work_per_thread,
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
            work_per_thread,
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
    cin = [
        scalar_arg(t.item(), input_casts[k] if input_casts else cute2torch[compute])
        if t.device.type == "cpu" and t.dim() == 0
        else _L.read_only(_iteration_view(t, plan.iteration, nout + k))
        for k, t in enumerate(inputs)
    ]
    cout = [_iteration_view(t, plan.iteration, k) for k, t in enumerate(storage_outs)]
    plan.fn(
        cin or None,
        cout,
        list(consts) or None,
        cutlass.Int32(len(plan.iteration.shape)),
        _L.stream(),
    )
    return tuple(outs)
