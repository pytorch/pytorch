# mypy: allow-untyped-defs
"""Shared CuTeDSL emission primitives for GEMM epilogues."""

import ast
import dataclasses
import math
import operator
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
)
from torch._inductor.ops_handler import ReductionType
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue_utils import normalize_shape


@dataclasses.dataclass(frozen=True)
class CuTeDSLEpilogueSchema:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalar_broadcast_names: frozenset[str]

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
    combine: Callable
    source: Callable
    finalize: Callable


def _identity_source(value):
    return value


def _square_source(value):
    return value * value


def _abs_source(value):
    import cutlass.cute as cute

    return cute.math.abs(value)


def _abs_scale_source(value):
    import cutlass.cute as cute

    return cute.math.max(cute.math.abs(value), cute.full_like(value, 1e-12)) / 448.0


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
    reduction_type: ReductionType,
    cute: Any,
    source_type: str = "identity",
    plan_type: str | None = None,
) -> MaterializedTensorSSAReduction:
    """Materialize the shared TensorSSA descriptor as CuTeDSL operands."""
    reduction = tensorssa_reduction(reduction_type)
    reduce_op = getattr(cute.ReductionOp, reduction.cute_op.rpartition(".")[2])
    combine = materialize_epilogue_function(
        f"def combine(lhs, rhs):\n    return {reduction.combine_expr}", cute
    )
    init_val = materialize_epilogue_function(
        f"def init():\n    return {reduction.init_val}", cute
    )()
    source = {
        "identity": _identity_source,
        "square": _square_source,
        "abs": _abs_source,
        "abs_scale": _abs_scale_source,
    }[source_type]
    finalize = (
        _mean_finalize
        if plan_type is not None and plan_type.startswith("mean")
        else _identity_finalize
    )
    return MaterializedTensorSSAReduction(
        reduce_op, init_val, combine, source, finalize
    )


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
            cute,
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
            args.reduction_algorithm,
            args.feeds_main,
        )

    def _primary_callbacks(self, *, include_consumer: bool = True) -> tuple[Any, ...]:
        reduction = self.reduction
        callbacks = (
            reduction.reduce_op,
            reduction.init_val,
            reduction.combine,
            reduction.source,
            reduction.finalize,
        )
        return (*callbacks, self.consumer) if include_consumer else callbacks

    def blockscaled_primary_constexprs(self) -> tuple[Any, ...]:
        args = self.args
        return (
            args.group,
            args.axis,
            args.feeds_main,
            *self._primary_callbacks(),
        )

    def constexprs(self, *, include_consumers: bool = True) -> tuple[Any, ...]:
        constexprs = (
            *self._common_constexprs(),
            *self._primary_callbacks(include_consumer=include_consumers),
        )
        return (
            (*constexprs, self.secondary_consumer) if include_consumers else constexprs
        )


def gemm_epilogue_cutedsl_op_name(target: Any) -> str | None:
    """Return the CuTeDSL operations-handler name for one FX target."""
    if isinstance(target, torch._ops.OpOverload):
        name = target.overloadpacket.__name__
    elif isinstance(target, str):
        name = target
    else:
        name = target.__name__ if callable(target) else None
    if name is not None:
        name = name.rsplit(".", 1)[-1]
    return "truediv" if name == "div" else name


def gemm_epilogue_arg(value: Any, env: dict[torch.fx.Node, Any], context: str) -> Any:
    """Translate FX references and constants into generated expressions."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported {context} epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return 'float("nan")'
        if math.isinf(value):
            return 'float("inf")' if value > 0 else 'float("-inf")'
        return value
    if isinstance(value, CuteDSLCSEVariable):
        return str(value)
    if isinstance(value, (str, torch.dtype)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(gemm_epilogue_arg(item, env, context) for item in value)
    raise NotImplementedError(f"unsupported {context} epilogue constant: {value!r}")


def gemm_epilogue_source_expr(value: Any) -> str:
    """Render a generated expression without quoting tuple components."""
    if isinstance(value, (tuple, list)):
        items = ", ".join(gemm_epilogue_source_expr(item) for item in value)
        return f"({items}{',' if len(value) == 1 else ''})"
    return str(value)


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    """Return the scalar value from an empty-shape ``aten.full`` node."""
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    if normalize_shape(node.args[0]) != ():
        return None
    value = node.args[1]
    return value if isinstance(value, (bool, int, float)) else None


def lower_gemm_epilogue_fx_call(
    node: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    context: str,
) -> Any:
    """Lower one FX call through the active CuTeDSL operations handler."""
    target = node.target
    op_name = gemm_epilogue_cutedsl_op_name(target)
    if op_name is None:
        raise NotImplementedError(f"unsupported {context} epilogue op: {target}")
    if op_name == "inline_asm_elementwise":
        kwargs = dict(kwargs)
        kwargs["asm"] = kwargs.pop("asm_str")
        input_values = []
        for input_node in node.args[: len(args)]:
            value = (
                input_node.meta.get("val")
                if isinstance(input_node, torch.fx.Node)
                else None
            )
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(
                    f"{context} inline asm inputs require tensor metadata"
                )
            input_values.append(value)
        kwargs["input_dtypes"] = tuple(value.dtype for value in input_values)
        kwargs["scalar_sources"] = tuple(
            all(isinstance(dim, int) and dim == 1 for dim in value.shape)
            for value in input_values
        )
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported {context} epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def lower_gemm_epilogue_fx_node(
    kernel: "GemmEpilogueCuteDSLKernel",
    env: dict[torch.fx.Node, Any],
    node: torch.fx.Node,
    *,
    context: str,
) -> Any:
    """Lower one ordinary FX expression through the shared CuTeDSL frontend."""
    if gemm_epilogue_cutedsl_op_name(node.target) in ("view", "reshape", "squeeze"):
        return gemm_epilogue_arg(node.args[0], env, context)
    if node.target is operator.getitem:
        source = gemm_epilogue_arg(node.args[0], env, context)
        index = node.args[1]
        if isinstance(source, (tuple, list)) and isinstance(index, int):
            return source[index]
    if (value := lower_full_scalar(node)) is not None:
        return gemm_epilogue_arg(value, env, context)
    args = tuple(gemm_epilogue_arg(arg, env, context) for arg in node.args)
    kwargs = {
        key: gemm_epilogue_arg(value, env, context)
        for key, value in node.kwargs.items()
    }
    with V.set_current_node(node):
        expression = lower_gemm_epilogue_fx_call(node, args, kwargs, context=context)
    if isinstance(expression, (tuple, list)):
        return expression
    meta = node.meta.get("val")
    dtype = meta.dtype if isinstance(meta, torch.Tensor) else torch.float32
    return kernel.cse.generate(kernel.body, expression, dtype=dtype, shape=(1,))


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
        result = self.newvar(bounds=bounds, dtype=dtype, shape=shape)
        body.writeline(f"{result} = {expr}")
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
