# mypy: allow-untyped-defs
"""Shared CuTeDSL emission primitives for GEMM epilogues."""

import ast
import dataclasses
from collections.abc import Callable
from typing import Any, cast

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    tensorssa_reduction,
)
from torch._inductor.kernel.gemm_epilogue import (
    GEMM_ACCUMULATOR_ARG_NAME,
    GemmReductionArguments,
    GemmReductionType,
)
from torch._inductor.ops_handler import ReductionType
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges


@dataclasses.dataclass(frozen=True)
class CuTeDSLEpilogueSchema:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalar_broadcast_names: frozenset[str]
    returns_local_reduce: bool = False

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in dict.fromkeys((*self.inputs, *self.outputs))
            if name != GEMM_ACCUMULATOR_ARG_NAME
        )


class CuTeDSLEpilogueSource(str):  # noqa: SLOT000
    schema: CuTeDSLEpilogueSchema

    def __new__(cls, source: str, schema: CuTeDSLEpilogueSchema):
        value = super().__new__(cls, source)
        value.schema = schema
        return value


def get_cutedsl_epilogue_schema(source: Any) -> CuTeDSLEpilogueSchema | None:
    return source.schema if isinstance(source, CuTeDSLEpilogueSource) else None


def gemm_epilogue_op_scope(
    cute: Any, *, mlir_math: Any | None = None
) -> dict[str, Any]:
    import operator

    import cutlass

    if mlir_math is None:
        from cutlass._mlir.dialects import math as default_mlir_math

        mlir_math = default_mlir_math

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


def materialize_epilogue_function(
    source: str, cute: Any, *, mlir_math: Any | None = None
) -> Any:
    function_names = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    if len(function_names) != 1:
        raise NotImplementedError("expected one GEMM epilogue function")
    scope = gemm_epilogue_op_scope(cute, mlir_math=mlir_math)
    exec(source, scope)
    return scope[function_names[0]]


@dataclasses.dataclass(frozen=True)
class MaterializedTensorSSAReduction:
    """CuTeDSL compile-time operands for a TensorSSA reduction."""

    reduce_op: object
    init_val: object
    combine: Callable | None
    source: Callable | None
    finalize: Callable | None


def _identity_finalize(value, group):
    return value


def _mean_finalize(value, group):
    return value / group


def canonical_tensorssa_reduction_type(reduction_type: str) -> ReductionType:
    """Return the TensorSSA primitive used by a reduction program."""
    if reduction_type == "mean":
        return "sum"
    return cast(ReductionType, reduction_type)


def materialize_tensorssa_reduction(
    reduction_type: GemmReductionType,
    cute: Any,
) -> MaterializedTensorSSAReduction:
    """Materialize the shared TensorSSA descriptor as CuTeDSL operands."""
    reduction = tensorssa_reduction(canonical_tensorssa_reduction_type(reduction_type))
    reduce_op = getattr(cute.ReductionOp, reduction.cute_op.rpartition(".")[2])
    combine = materialize_epilogue_function(
        f"def combine(lhs, rhs):\n    return {reduction.combine_expr}", cute
    )
    init_val = materialize_epilogue_function(
        f"def init():\n    return {reduction.init_val}", cute
    )()
    finalize = _mean_finalize if reduction_type == "mean" else _identity_finalize
    return MaterializedTensorSSAReduction(reduce_op, init_val, combine, None, finalize)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GemmReductionCompileConfig:
    """Materialized compile-time reduction arguments shared by NVGEMM kernels."""

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

        if args.tensor_epilogue_returns_local_reduce:
            if any(
                selector is not None
                for selector in (
                    args.reduction_type,
                    args.source_fn,
                )
            ):
                raise RuntimeError(
                    "generated tensor reductions cannot also specify reduction_type or source_fn"
                )
            if args.geometry.needs_physical_callbacks != (args.combine_fn is not None):
                raise RuntimeError(
                    "combine_fn must be present exactly when a generated reduction "
                    "crosses TensorSSA fragments"
                )
            reduction = MaterializedTensorSSAReduction(
                None,
                0.0,
                materialize(args.combine_fn),
                None,
                materialize(args.finalizer_fn),
            )
        else:
            if args.reduction_type is None or args.source_fn is None:
                raise RuntimeError(
                    "kernel-driven GEMM reductions require reduction_type and source_fn"
                )
            reduction = materialize_tensorssa_reduction(
                args.reduction_type,
                cute,
            )
            reduction = dataclasses.replace(
                reduction, source=materialize(args.source_fn)
            )
            finalizer = materialize(args.finalizer_fn)
            if finalizer is not None:
                reduction = dataclasses.replace(reduction, finalize=finalizer)

        consumer_finalizer = materialize(args.consumer_finalizer_fn)

        def materialize_consumer(source: str | None) -> Any:
            consumer = materialize(source)
            if consumer is None or consumer_finalizer is None:
                return consumer

            def consume(accumulator, primary_reduction, secondary_reduction):
                return consumer(
                    accumulator,
                    consumer_finalizer(primary_reduction, args.group),
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
            args.feeds_main,
            args.tensor_epilogue_returns_local_reduce,
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

    def blockscaled_primary_constexprs(self) -> tuple[Any, ...]:
        args = self.args
        return (
            args.group,
            args.axis,
            args.feeds_main,
            args.tensor_epilogue_returns_local_reduce,
            *self._primary_callbacks(),
        )

    def constexprs(self) -> tuple[Any, ...]:
        return (
            *self._common_constexprs(),
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
        self._cache: dict[str, CuteDSLCSEVariable] = {}

    def newvar(self, *, bounds=None, dtype=None, shape=None) -> CuteDSLCSEVariable:
        name = f"tmp{self.index}"
        self.index += 1
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )

    def put(self, cache_key: str, value: CuteDSLCSEVariable) -> None:
        self._cache[cache_key] = value

    def try_get(self, cache_key: str) -> CuteDSLCSEVariable | None:
        return self._cache.get(cache_key)

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        cached = self.try_get(expr)
        if cached is not None:
            return cached
        result = self.newvar(bounds=bounds, dtype=dtype, shape=shape)
        body.writeline(f"{result} = {expr}")
        self.put(expr, result)
        return result


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
