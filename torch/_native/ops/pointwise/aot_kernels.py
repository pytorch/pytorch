"""Flat, dynamic-shape C ABI for the shared pointwise TMA executor."""

from typing import Any

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from . import kernel, ops
from ._limits import _MIN_ALIGNMENT, _THREADS_PER_BLOCK, _TMA_ALIGNMENT, _TMA_BOX_SIDE


class _AotTma:
    def __init__(
        self,
        fn: str,
        dtype: Any,
        mask: tuple[bool | None, ...],
        out_type: Any,
        vector_bytes: int,
        vector_ept: int,
    ) -> None:
        self.comp_type = cutlass.Float32
        box, block = _TMA_BOX_SIDE, _THREADS_PER_BLOCK
        if vector_bytes != 0:
            box = max(1, vector_bytes // (dtype.width // 8))
        config = kernel._TmaConfig(mask, box, block, (), ((), ()), vector_ept)
        executor = (
            kernel._ElementwiseScalarVector
            if vector_bytes != 0
            else kernel._ElementwiseTma2D
        )
        self.op = executor(
            fn=ops.get_fn(fn),
            ninputs=len(mask),
            nconsts=1 if fn in ("_add", "_sub") else 0,
            comp_type=self.comp_type,
            out_type=out_type,
            in_type=dtype,
            config=config,
        )

    @cute.jit
    def binary(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        out: cute.Tensor,
        alpha: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.op(
            [x, y],
            out,
            self.comp_type(alpha),
            self.comp_type(1),
            self.comp_type(1),
            stream,
        )

    @cute.jit
    def unary(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        alpha: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.op(
            [x],
            out,
            self.comp_type(alpha),
            self.comp_type(1),
            self.comp_type(1),
            stream,
        )


def build(spec: dict[str, Any]) -> dict[str, Any]:
    types = {
        "float16": "Float16",
        "bfloat16": "BFloat16",
        "float32": "Float32",
    }
    dtype = getattr(cutlass, types[spec["dtype"]])
    scalar_dtype = (
        getattr(cutlass, types[spec["scalar_dtype"]]) if spec["scalar_dtype"] else dtype
    )
    mask = tuple(spec["transposed"])
    vector_bytes = spec["vector_bytes"]
    output_type = (
        cutlass.Int8 if spec["boolean"] else getattr(cutlass, types[spec["out_dtype"]])
    )
    op = _AotTma(spec["fn"], dtype, mask, output_type, vector_bytes, spec["vector_ept"])
    box = op.op.box
    shapes = (
        (cute.sym_int(),)
        if vector_bytes != 0
        else (cute.sym_int(divisibility=box), cute.sym_int(divisibility=box))
    )
    fakes = []
    tensor_args: list[dict[str, Any]] = []
    names = ("x",) if len(mask) == 1 else ("x", "y")
    for name, transposed in zip(names, mask):
        if transposed is None:
            fakes.append(
                cute.runtime.make_fake_tensor(
                    scalar_dtype,
                    (1,),
                    stride=(1,),
                    assumed_align=scalar_dtype.width // 8,
                )
            )
            tensor_args.append({"name": name, "read_only": True})
            continue
        pitch = (
            box
            if vector_bytes != 0
            else cute.sym_int64(divisibility=_MIN_ALIGNMENT // (dtype.width // 8))
        )
        stride = (1,) if vector_bytes != 0 else (1, pitch) if transposed else (pitch, 1)
        fakes.append(
            cute.runtime.make_fake_tensor(
                dtype,
                shapes,
                stride=stride,
                assumed_align=max(vector_bytes, dtype.width // 8)
                if vector_bytes != 0
                else _TMA_ALIGNMENT
                if transposed
                else _MIN_ALIGNMENT,
            )
        )
        tensor_args.append(
            {
                "name": name,
                "dynamic_sizes": [0] if vector_bytes != 0 else [0, 1],
                "dynamic_strides": []
                if vector_bytes != 0
                else [1 if transposed else 0],
                "read_only": True,
            }
        )
    fakes.append(
        cute.runtime.make_fake_tensor(
            output_type,
            shapes if vector_bytes != 0 else (1, *shapes),
            stride=(1,)
            if vector_bytes != 0
            else (0, cute.sym_int64(divisibility=box), 1),
            assumed_align=box * output_type.width // 8
            if vector_bytes != 0
            else _MIN_ALIGNMENT,
        )
    )
    tensor_args.append(
        {
            "name": "out",
            "dynamic_sizes": [0] if vector_bytes != 0 else [1, 2],
            "dynamic_strides": [] if vector_bytes != 0 else [1],
        }
    )
    tag = "".join("s" if value is None else "t" if value else "n" for value in mask)
    if not spec["boolean"] and spec["out_dtype"] != spec["dtype"]:
        tag += f"_out_{spec['out_dtype']}"
    if spec["scalar_dtype"]:
        tag += f"_{spec['scalar_dtype']}"
    if vector_bytes != 0:
        tag += f"_vec{vector_bytes}_ept{spec['vector_ept']}"
    return {
        "kind": "cutedsl",
        "fn": op.unary if len(mask) == 1 else op.binary,
        "prefix": f"pointwise_{spec['fn'].lstrip('_')}_{spec['dtype']}_{tag}",
        "fake_args": [*fakes, cutlass.Float32(1), cute.runtime.make_fake_stream()],
        "tensor_args": tensor_args,
        "scalar_args": [{"name": "alpha", "ctype": "float"}],
    }
