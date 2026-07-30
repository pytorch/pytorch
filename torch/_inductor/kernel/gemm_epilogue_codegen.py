# mypy: allow-untyped-defs
"""Shared CuTeDSL expression emission for GEMM epilogues."""

from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
)
from torch._inductor.ir import ComputedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue_ir import GemmEpilogueIRAnalysis, GemmEpilogueIRExpression


def gemm_epilogue_op_scope(cute: Any) -> dict[str, Any]:
    import operator

    import cutlass
    from cutlass._mlir.dialects import math as mlir_math

    def sigmoid(x: Any) -> Any:
        return 1.0 / (1.0 + cute.math.exp(-x))

    return {
        "cutlass": cutlass,
        "operator": operator,
        "mlir_math": mlir_math,
        "cute": cute,
        "erf": cute.math.erf,
        "exp": cute.math.exp,
        "gelu": lambda x: 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476)),
        "relu": lambda x: cute.math.max(x, cute.full_like(x, 0.0)),
        "sigmoid": sigmoid,
        "silu": lambda x: x * sigmoid(x),
        "tanh": cute.math.tanh,
    }


class GemmEpilogueCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class GemmEpilogueCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class GemmEpilogueCuteDSLKernel:
    def __init__(self) -> None:
        self.body = GemmEpilogueCuteDSLBody()
        self.cse = GemmEpilogueCuteDSLCSE()


class GemmEpilogueCuteDSLOpOverrides(CuteDSLOpOverrides):
    """Normalize ATen spellings before emitting CuTeDSL expressions."""

    # Aten add/sub carry alpha as schema sugar; CuTeDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                f"unsupported GEMM epilogue _to_copy kwargs: {unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)


class GemmEpilogueIRCodegen:
    """Lower scheduler loop-body expressions to a CuTeDSL fragment function."""

    def __init__(self, accumulator: str, removed_buffers: OrderedSet[str]) -> None:
        self.accumulator = accumulator
        self.removed_buffers = removed_buffers
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.reads: OrderedSet[str] = OrderedSet()
        self.input_values: dict[str, CuteDSLCSEVariable] = {}
        self.accumulator_value = CuteDSLCSEVariable(
            "accum", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )

    def load(self, name: str, stored: Any) -> Any:
        if stored is not None:
            return self.lower(stored)
        if name == self.accumulator:
            return self.accumulator_value
        self.reads.add(name)
        if name not in self.input_values:
            dtype = V.graph.get_dtype(name)
            self.input_values[name] = CuteDSLCSEVariable(
                name, ValueRanges.unknown(), dtype=dtype, shape=(1,)
            )
        return self.input_values[name]

    def lower(self, value: Any) -> Any:
        if not isinstance(value, GemmEpilogueIRExpression):
            if isinstance(value, tuple):
                return tuple(self.lower(item) for item in value)
            return value
        if value.op == "load":
            name, _, stored = value.args
            return self.load(name, stored)
        if value.op == "identity":
            return self.lower(value.args[0])
        if value.op in ("reduction", "to_dtype_bitcast"):
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue does not support {value.op}"
            )
        lowered_args = tuple(self.lower(arg) for arg in value.args)
        lowered_kwargs = {key: self.lower(arg) for key, arg in value.kwargs}
        op = getattr(GemmEpilogueCuteDSLOpOverrides, value.op, None)
        if op is None:
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue op is not implemented: {value.op}"
            )
        return op(*lowered_args, **lowered_kwargs)

    def render(
        self, buffers: Sequence[ComputedBuffer], fn_name: str
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        outputs: list[tuple[str, Any]] = []
        if self.accumulator not in self.removed_buffers:
            outputs.append((self.accumulator, self.accumulator_value))
        for name, store in analysis.stores.items():
            if name not in self.removed_buffers:
                outputs.append((name, self.lower(store.value)))
        if not outputs:
            raise NotImplementedError("CuTeDSL GEMM epilogue has no outputs")

        renames = {name: name for name in self.reads}
        result_names: list[str] = []
        for index, (buffer_name, value) in enumerate(outputs):
            result_name = "D" if index == len(outputs) - 1 else f"output{index}"
            self.kernel.body.writeline(f"{result_name} = {value}")
            renames[result_name] = buffer_name
            result_names.append(result_name)

        params = ", ".join(("accum", *self.reads))
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        source = (
            f"def {fn_name}({params}):\n{body}\n    return {', '.join(result_names)}"
        )
        return list(self.reads), [name for name, _ in outputs], renames, source

    @classmethod
    def from_buffers(
        cls,
        accumulator: str,
        buffers: Sequence[ComputedBuffer],
        removed_buffers: OrderedSet[str],
        fn_name: str,
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        codegen = cls(accumulator, removed_buffers)
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            return codegen.render(buffers, fn_name)
