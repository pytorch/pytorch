"""Flat, dynamic-shape C ABI for the shared pointwise TMA executor."""

from typing import Any

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from . import kernel, ops
from ._limits import _MIN_ALIGNMENT, _THREADS_PER_BLOCK, _TMA_ALIGNMENT, _TMA_BOX_SIDE


class _AotTma:
    def __init__(
        self, fn: str, dtype: Any, mask: tuple[bool, ...], boolean: bool
    ) -> None:
        self.comp_type = cutlass.Float32
        box, block = _TMA_BOX_SIDE, _THREADS_PER_BLOCK
        config = kernel._TmaConfig(mask, box, block, (), ((), ()))
        self.op = kernel._ElementwiseTma2D(
            fn=ops.get_fn(fn),
            ninputs=len(mask),
            nconsts=1 if fn in ("_add", "_sub") else 0,
            comp_type=self.comp_type,
            out_type=cutlass.Int8 if boolean else dtype,
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
    dtype = getattr(
        cutlass,
        {
            "float16": "Float16",
            "bfloat16": "BFloat16",
            "float32": "Float32",
        }[spec["dtype"]],
    )
    mask = tuple(spec["transposed"])
    op = _AotTma(spec["fn"], dtype, mask, spec["boolean"])
    box = op.op.box
    shapes = (cute.sym_int(divisibility=box), cute.sym_int(divisibility=box))
    fakes = []
    tensor_args = []
    names = ("x",) if len(mask) == 1 else ("x", "y")
    for name, transposed in zip(names, mask):
        pitch = cute.sym_int64(divisibility=_MIN_ALIGNMENT // (dtype.width // 8))
        stride = (1, pitch) if transposed else (pitch, 1)
        fakes.append(
            cute.runtime.make_fake_tensor(
                dtype,
                shapes,
                stride=stride,
                assumed_align=_TMA_ALIGNMENT if transposed else _MIN_ALIGNMENT,
            )
        )
        tensor_args.append(
            {
                "name": name,
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [1 if transposed else 0],
                "read_only": True,
            }
        )
    output_type = cutlass.Int8 if spec["boolean"] else dtype
    fakes.append(
        cute.runtime.make_fake_tensor(
            output_type,
            (1, *shapes),
            stride=(0, cute.sym_int64(divisibility=box), 1),
            assumed_align=_MIN_ALIGNMENT,
        )
    )
    tensor_args.append({"name": "out", "dynamic_sizes": [1, 2], "dynamic_strides": [1]})
    tag = "".join("t" if value else "n" for value in mask)
    return {
        "kind": "cutedsl",
        "fn": op.unary if len(mask) == 1 else op.binary,
        "prefix": f"pointwise_{spec['fn'].lstrip('_')}_{spec['dtype']}_{tag}",
        "fake_args": [*fakes, cutlass.Float32(1), cute.runtime.make_fake_stream()],
        "tensor_args": tensor_args,
        "scalar_args": [{"name": "alpha", "ctype": "float"}],
    }
