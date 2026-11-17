from typing import Any

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from ...cutedsl import launch
from ...cutedsl.aot import export_args
from ...cutedsl.config import POINTWISE_TMA_SHAPES
from ...cutedsl.dtypes import torch2cute
from . import ops
from ._packing import flat_copy, FLAT_EPT, FLAT_THREADS
from .kernel import _ElementwiseVec, _fake_vec
from .layout import TmaLayout
from .table import POINTWISE_DEF_TABLE
from .tma import ElementwiseTma
from .transpose_copy import _CUTE_WORD, tiled_kernel


class _Arithmetic:
    def __init__(self, op: _ElementwiseVec | ElementwiseTma) -> None:
        self.op: _ElementwiseVec | ElementwiseTma = op


class _Arithmetic32(_Arithmetic):
    @cute.jit
    def __call__(
        self,
        x0: cute.Tensor,
        x1: cute.Tensor,
        x2: cute.Tensor,
        y: cute.Tensor,
        c0: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        op = self.op
        op([x0, x1, x2][: op.nin], [y], (c0,)[: op.nconsts], stream)


class _Arithmetic64(_Arithmetic):
    @cute.jit
    def __call__(
        self,
        x0: cute.Tensor,
        x1: cute.Tensor,
        x2: cute.Tensor,
        y: cute.Tensor,
        c0: cutlass.Float64,
        stream: cuda.CUstream,
    ) -> None:
        op = self.op
        op([x0, x1, x2][: op.nin], [y], (c0,)[: op.nconsts], stream)


def build(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    if spec["op"] == "copy_":
        return _build_copy(spec)

    row = next(r for r in POINTWISE_DEF_TABLE if r.aten == spec["op"])
    storage = torch2cute[getattr(torch, spec["dtype"])]
    if len(row.scalars) > 1:
        raise ValueError("AOT arithmetic supports one scalar argument")
    signature = row.fn.signature(
        ((getattr(torch, spec["dtype"]), False),) * row.nin,
        row.promotion,
        row.nout,
    )
    compute = torch2cute[signature.compute]
    casts = (torch2cute[signature.storage],) * row.nin
    results = tuple(torch2cute[d] for d in signature.results)
    if spec["layout"] == "dense":
        vector = spec["vec_bits"] // compute.width
        op = _ElementwiseVec(
            ops.get_fn(row.fn),
            row.nin,
            1,
            len(row.scalars),
            compute,
            [storage],
            casts,
            ept=spec["ept"],
            vector=vector,
            result_types=results,
        )
        tensors = [_fake_vec(storage, vector) for _ in range(4)]
        tag = f"v{spec['vec_bits']}_e{spec['ept']}"
    else:
        orientations = tuple(bool(spec["transpose"] & (1 << k)) for k in range(row.nin))
        shape = spec.get("shape", 0)
        rows, cols = (
            POINTWISE_TMA_SHAPES[shape - 1]
            if shape
            else (launch.sym(32), launch.sym(32))
        )
        views = tuple(
            (
                (rows, cols),
                (1, cute.sym_int64(divisibility=16 // (storage.width // 8)))
                if t
                else (cute.sym_int64(divisibility=16 // (storage.width // 8)), 1),
            )
            for t in orientations
        )
        first = sum(orientations) > row.nin - sum(orientations) + 1
        out_strides = (
            (rows * cols, cols, 1) if shape else (cute.sym_int64(), cute.sym_int64(), 1)
        )
        layout = TmaLayout(orientations, views, ((1, rows, cols), out_strides), first)
        op = ElementwiseTma(
            ops.get_fn(row.fn),
            len(row.scalars),
            compute,
            storage,
            storage,
            layout,
            casts,
            results,
        )
        inputs, outputs = op.fake_args(layout, (storage,) * row.nin)
        tensors = inputs + [inputs[0]] * (3 - row.nin) + outputs
        tag = f"t{spec['transpose']}" + (f"_s{shape}" if shape else "")
    prefix = spec["op"].replace(".", "_")
    fn = (_Arithmetic64 if compute is cutlass.Float64 else _Arithmetic32)(op)
    return export_args(
        fn,
        *tensors,
        prefix=f"pointwise_{prefix}_{spec['dtype']}_{tag}",
        read_only=("x0", "x1", "x2"),
    )


class _RuntimeCopy:
    def __init__(self, normalize: bool, pack: int = 1) -> None:
        self.normalize: bool = normalize
        self.pack: int = pack

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        b: cutlass.Int32,
        middle: cutlass.Int32,
        a: cutlass.Int32,
        inner: cutlass.Int32,
        numel: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        geometry = (b, middle, a, inner, numel)
        flat_copy(src, dst, geometry, 8 // self.pack, self.pack, self.normalize).launch(
            grid=[cute.ceil_div(cute.size(dst), FLAT_THREADS * FLAT_EPT), 1, 1],
            block=[FLAT_THREADS, 1, 1],
            stream=stream,
        )


def _build_copy(spec: dict[str, Any]) -> dict[str, Any]:
    from .aot_copy import Copy

    if spec["layout"] not in Copy.LAYOUTS:
        raise ValueError(f"unknown transpose AOT layout: {spec['layout']}")
    width, lanes = spec["word"], spec["lanes"]
    dtype = _CUTE_WORD[width]
    if spec["layout"] in ("payload", "packed"):
        tensors = [
            launch.fake_compact(dtype, (launch.sym(),), align=width) for _ in range(2)
        ]
        op = _RuntimeCopy(spec["normalize"], spec["pack"])
    else:
        op, tensors = tiled_kernel(
            width,
            lanes,
            spec["normalize"],
            spec["full"],
            spec["layout"] == "vector",
            tma=spec["layout"] == "tma",
        )
    prefix = f"transpose_{spec['layout']}_w{width}_l{spec['lanes']}_n{int(spec['normalize'])}_f{int(spec['full'])}_p{spec['pack']}"
    return export_args(op, *tensors, prefix=prefix, read_only=("src",))
