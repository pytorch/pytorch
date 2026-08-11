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
from torch._inductor.kernel.gemm_epilogue import GemmReductionArguments
from torch._inductor.virtualized import V
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class GemmReductionCompileConfig:
    args: GemmReductionArguments
    reduction: Any
    consumer: Any
    secondary_consumer: Any

    @classmethod
    def from_args(
        cls, args: GemmReductionArguments, cute: Any
    ) -> "GemmReductionCompileConfig":
        def materialize(source: str | None) -> Any:
            return (
                None if source is None else materialize_epilogue_function(source, cute)
            )

        reduction = materialize_tensorssa_reduction(
            canonical_tensorssa_reduction_type(args.reduction_type),
            args.source_type,
            args.reduction_type,
        )
        finalizer = materialize(args.finalizer_fn)
        if finalizer is not None:
            reduction = dataclasses.replace(reduction, finalize=finalizer)

        def materialize_consumer(source: str | None) -> Any:
            consumer = materialize(source)
            if consumer is None or finalizer is not None:
                return consumer

            def consume(accumulator, primary_reduction, secondary_reduction):
                return consumer(
                    accumulator,
                    reduction.finalize(primary_reduction, args.group),
                    secondary_reduction,
                )

            return consume

        return cls(
            args=args,
            reduction=reduction,
            consumer=materialize_consumer(args.consumer_fn),
            secondary_consumer=materialize_consumer(args.secondary_consumer_fn),
        )

    def _common_constexprs(self) -> tuple[Any, ...]:
        args = self.args
        return (
            args.group,
            args.axis,
            args.reduction_type,
            args.source_type,
            args.feeds_main,
        )

    def _primary_callbacks(self) -> tuple[Any, ...]:
        reduction = self.reduction
        return (
            reduction.reduce_op,
            reduction.init_val,
            reduction.combine,
            reduction.source,
            reduction.finalize,
            self.consumer,
        )

    def primary_constexprs(self) -> tuple[Any, ...]:
        return self._common_constexprs() + self._primary_callbacks()

    def blockscaled_primary_constexprs(self) -> tuple[Any, ...]:
        args = self.args
        return (
            args.group,
            args.axis,
            args.feeds_main,
            *self._primary_callbacks(),
        )

    def constexprs(self) -> tuple[Any, ...]:
        return (
            *self._common_constexprs(),
            self.args.secondary_feed_type,
            *self._primary_callbacks(),
            self.secondary_consumer,
        )


class GemmEpilogueCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class GemmEpilogueCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0
        self.cache: dict[str, CuteDSLCSEVariable] = {}

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        if expr in self.cache:
            return self.cache[expr]
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        value = CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )
        self.cache[expr] = value
        return value


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
    def where(condition: Any, a: Any, b: Any) -> Any:
        """Preserve all-scalar conditionals as scalar epilogue expressions."""
        if any(
            CuteDSLOpOverrides._is_tensor_like(value) for value in (condition, a, b)
        ):
            return CuteDSLOpOverrides.where(condition, a, b)
        result_expr = (
            f"({CuteDSLOpOverrides._as_expr(a)} if "
            f"{CuteDSLOpOverrides._as_expr(condition)} else "
            f"{CuteDSLOpOverrides._as_expr(b)})"
        )
        cse_vars = tuple(
            CuteDSLOpOverrides._get_cse_var(value) for value in (a, b, condition)
        )
        if all(value is None for value in cse_vars):
            return result_expr
        dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b, condition)
        result = V.kernel.cse.generate(
            V.kernel.body,
            result_expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else torch.int32,
        )
        result.is_scalar_expr = True
        return result

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
