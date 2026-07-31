# mypy: allow-untyped-defs
"""Shared CuTeDSL emission primitives for GEMM epilogues."""

import ast
import dataclasses
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    canonical_tensorssa_reduction_type,
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    materialize_tensorssa_reduction,
)
from torch.utils._sympy.value_ranges import ValueRanges


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


def materialize_epilogue_function(source: str, cute: Any) -> Any:
    function_names = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    if len(function_names) != 1:
        raise NotImplementedError("expected one GEMM epilogue function")
    scope = gemm_epilogue_op_scope(cute)
    exec(source, scope)
    return scope[function_names[0]]


def with_reduction_finalizer(reduction: Any, source: str | None, cute: Any) -> Any:
    if source is None:
        return reduction
    return dataclasses.replace(
        reduction, finalize=materialize_epilogue_function(source, cute)
    )


def materialize_reduction_consumer(source: str | None, cute: Any) -> Any:
    if source is None:
        return None
    return materialize_epilogue_function(source, cute)


def materialize_reduction_callbacks(args: Any, cute: Any) -> tuple[Any, Any, Any]:
    reduction = materialize_tensorssa_reduction(
        canonical_tensorssa_reduction_type(args.reduction_type),
        args.source_type,
        args.reduction_type,
    )
    reduction = with_reduction_finalizer(reduction, args.finalizer_fn, cute)
    return (
        reduction,
        materialize_reduction_consumer(args.consumer_fn, cute),
        materialize_reduction_consumer(args.secondary_consumer_fn, cute),
    )


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
