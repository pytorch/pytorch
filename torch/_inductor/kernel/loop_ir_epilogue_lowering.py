# mypy: allow-untyped-defs
"""Lower GEMM epilogue loop IR to shared epilogue contracts."""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.ir import ComputedBuffer
from torch._inductor.kernel.gemm_epilogue import GemmReductionConfig
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRExpression:
    """Operation captured from a lowered GEMM epilogue loop body.

    Attributes:
        op: Virtualized Inductor operation name.
        args: Captured positional arguments.
        kwargs: Captured keyword arguments in deterministic key order.
    """

    op: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()
    loads: frozenset[str] = frozenset()
    reductions: tuple["GemmEpilogueIRReduction", ...] = ()


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRReduction:
    reduction_type: str
    source: GemmEpilogueIRExpression
    result: int | None = None
    source_type: str | None = None


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRRegion:
    output_name: str
    reductions: tuple[GemmEpilogueIRReduction, ...]
    expression: GemmEpilogueIRExpression

    @property
    def algorithm(self) -> str:
        reduction_types = OrderedSet(
            reduction.reduction_type for reduction in self.reductions
        )
        if "online_softmax_reduce" in reduction_types:
            return "online_softmax"
        if "welford_reduce" in reduction_types:
            return "welford"
        return "generic"


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRStore:
    """Symbolic store produced by replaying a lowered epilogue loop body.

    Attributes:
        index: Symbolic destination index used by the lowered store.
        value: Captured expression written at that index.
    """

    index: sympy.Expr
    value: GemmEpilogueIRExpression


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIROutputRole:
    """Transitive inputs that determine one captured epilogue output.

    Attributes:
        transitive_inputs: All buffers loaded directly or through stored values.
        reduction_inputs: Loaded buffers whose captured values contain reductions.
    """

    transitive_inputs: frozenset[str]
    reduction_inputs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRFinalizer:
    """Normalized operation applied to a completed grouped reduction."""

    output_name: str
    source_name: str
    kind: str


def _expression_values(value: Any):
    if isinstance(value, GemmEpilogueIRExpression):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _expression_values(item)


def _unique_reductions(
    values: Sequence[GemmEpilogueIRExpression],
) -> tuple[GemmEpilogueIRReduction, ...]:
    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for value in values:
        for reduction in value.reductions:
            if id(reduction) not in seen:
                seen.add(id(reduction))
                reductions.append(reduction)
    return tuple(reductions)


class _GemmEpilogueIRHandler(DefaultHandler):
    def __init__(self) -> None:
        self.stores: dict[str, GemmEpilogueIRStore] = {}

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> GemmEpilogueIRExpression:
        values = tuple(_expression_values((*args, *tuple(kwargs.values()))))
        return GemmEpilogueIRExpression(
            name,
            args,
            tuple(sorted(kwargs.items())),
            frozenset().union(*(value.loads for value in values)),
            _unique_reductions(values),
        )

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        return sympy.Symbol(f"indirect_{len(self.stores)}", integer=True)

    def load(self, name: str, index: sympy.Expr) -> GemmEpilogueIRExpression:
        stored = self.stores.get(name)
        if stored is None:
            return GemmEpilogueIRExpression(
                "load", (name, index, None), loads=frozenset((name,))
            )
        return GemmEpilogueIRExpression(
            "load",
            (name, index, stored.value),
            loads=frozenset((name,)) | stored.value.loads,
            reductions=stored.value.reductions,
        )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        args = (dtype, src_dtype, reduction_type, value)
        if reduction_type in ("online_softmax_reduce", "welford_reduce"):
            count = 2 if reduction_type == "online_softmax_reduce" else 3
            return tuple(
                GemmEpilogueIRExpression(
                    "reduction",
                    (*args, index),
                    loads=value.loads,
                    reductions=(
                        *value.reductions,
                        GemmEpilogueIRReduction(reduction_type, value, index),
                    ),
                )
                for index in range(count)
            )
        return GemmEpilogueIRExpression(
            "reduction",
            args,
            loads=value.loads,
            reductions=(
                *value.reductions,
                GemmEpilogueIRReduction(reduction_type, value),
            ),
        )

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: GemmEpilogueIRExpression,
        mode=None,
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)

    def store_reduction(
        self, name: str, index: sympy.Expr, value: GemmEpilogueIRExpression
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)


def _loaded_names(expr: Any) -> frozenset[str]:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return frozenset()
    if expr.op == "load":
        name, _, stored = expr.args
        return frozenset((name,)) | _loaded_names(stored)
    return frozenset().union(*(_loaded_names(arg) for arg in expr.args))


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRAnalysis:
    """Candidate-wide semantic view of existing lowered loop bodies."""

    stores: dict[str, GemmEpilogueIRStore]

    @classmethod
    def from_buffers(
        cls, buffers: Sequence[ComputedBuffer]
    ) -> "GemmEpilogueIRAnalysis":
        handler = _GemmEpilogueIRHandler()
        with V.set_ops_handler(handler):
            for buffer in buffers:
                buffer.get_store_function()(*buffer.data.inner_fn_args())
        return cls(handler.stores)

    @classmethod
    def store_from_buffer(cls, buffer: ComputedBuffer) -> GemmEpilogueIRStore | None:
        return cls.from_buffers((buffer,)).store(buffer.get_name())

    def store(self, name: str) -> GemmEpilogueIRStore | None:
        return self.stores.get(name)

    def output_role(self, name: str) -> GemmEpilogueIROutputRole | None:
        store = self.store(name)
        if store is None:
            return None
        transitive_inputs = store.value.loads or _loaded_names(store.value)
        return GemmEpilogueIROutputRole(
            transitive_inputs,
            transitive_inputs & self.reduction_stores,
        )

    def grouped_reduction(
        self,
        output_name: str,
        source_name: str,
        group: int,
        axis: int,
        source_dtype: torch.dtype,
    ) -> GemmReductionConfig | None:
        store = self.store(output_name)
        classified = (
            grouped_reduction_ir(store, source_name, group, source_dtype)
            if store is not None
            else None
        )
        if classified is None:
            return None
        reduction_type, source_type = classified
        return GemmReductionConfig(
            output_name, group, axis, reduction_type, source_type
        )

    def reduction_region(
        self,
        output_name: str,
        source_name: str,
        group: int,
        source_dtype: torch.dtype,
    ) -> GemmEpilogueIRRegion | None:
        store = self.store(output_name)
        if store is None:
            return None
        reductions = store.value.reductions or _synthetic_reductions_ir(
            store.value, store.index, source_name, group, source_dtype
        )
        if not reductions:
            return None
        if any(
            (reduction.source.loads or _loaded_names(reduction.source))
            != frozenset((source_name,))
            for reduction in reductions
        ):
            return None
        return GemmEpilogueIRRegion(output_name, reductions, store.value)

    def reduction_finalizer(
        self,
        output_name: str,
        source_name: str,
        group: int | None = None,
    ) -> GemmEpilogueIRFinalizer | None:
        store = self.store(output_name)
        if store is None:
            return None
        if operation_names_ir(store).issubset(
            ("load", "to_dtype", "to_dtype_bitcast", "identity")
        ):
            kind = "identity"
        elif group is not None and single_source_affine_ir(store, source_name) == (
            1.0 / group,
            0.0,
        ):
            kind = "mean"
        elif is_absmax_scale_finalizer_ir(store, source_name):
            kind = "absmax_scale"
        elif (
            store.value.loads == frozenset((source_name,))
            and not store.value.reductions
        ):
            kind = "generic"
        else:
            return None
        return GemmEpilogueIRFinalizer(output_name, source_name, kind)

    @property
    def reduction_stores(self) -> frozenset[str]:
        return frozenset(
            name
            for name, store in self.stores.items()
            if store.value.reductions or _contains_reduction(store.value)
        )


def _constant_value(expr: Any) -> Any | None:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op in ("constant", "index_expr") and expr.args:
        return expr.args[0]
    if expr.op in ("to_dtype", "to_dtype_bitcast") and expr.args:
        return _constant_value(expr.args[0])
    return None


def _strip_conversions(expr: Any) -> Any:
    while (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op in ("to_dtype", "to_dtype_bitcast", "identity")
        and expr.args
    ):
        expr = expr.args[0]
    return expr


def _walk(expr: Any):
    if not isinstance(expr, GemmEpilogueIRExpression):
        return
    yield expr
    for arg in expr.args:
        yield from _walk(arg)


def grouped_reduction_axis_ir(
    reduction: GemmEpilogueIRReduction, group: int, n: int
) -> int | None:
    """Infer grouped M/N geometry from the reduction source load strides."""
    indices = [
        expr.args[1]
        for expr in _walk(reduction.source)
        if expr.op == "load" and len(expr.args) > 1
    ]
    unique_indices = []
    for index in indices:
        if not any(sympy.simplify(index - other) == 0 for other in unique_indices):
            unique_indices.append(index)
    if len(unique_indices) != group:
        return None

    def is_progression(step: int) -> bool:
        return any(
            all(
                any(
                    sympy.simplify(index - (base + offset * step)) == 0
                    for index in unique_indices
                )
                for offset in range(group)
            )
            for base in unique_indices
        )

    axes = [axis for axis, step in ((1, 1), (0, n)) if is_progression(step)]
    return axes[0] if len(axes) == 1 else None


def operation_names_ir(store: GemmEpilogueIRStore) -> frozenset[str]:
    return frozenset(expr.op for expr in _walk(store.value))


def _source_transform(
    expr: Any,
    source_name: str,
    allowed_conversion_dtypes: frozenset[torch.dtype] | None = None,
) -> str | None:
    if allowed_conversion_dtypes is None:
        expr = _strip_conversions(expr)
    else:
        while isinstance(expr, GemmEpilogueIRExpression) and expr.args:
            if expr.op == "identity":
                expr = expr.args[0]
            elif expr.op == "to_dtype":
                if len(expr.args) < 2 or expr.args[1] not in allowed_conversion_dtypes:
                    return None
                expr = expr.args[0]
            elif expr.op == "to_dtype_bitcast":
                return None
            else:
                break
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        return "identity" if expr.args[0] == source_name else None
    if expr.op == "abs":
        return (
            "abs"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "mul" and expr.args[0] == expr.args[1]:
        return (
            "square"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "pow":
        exponent = _constant_value(expr.args[1])
        return (
            "square"
            if exponent == 2
            and _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    return None


def _flatten_associative(expr: Any, op: str) -> list[Any]:
    stripped = _strip_conversions(expr)
    if isinstance(stripped, GemmEpilogueIRExpression) and stripped.op == op:
        return _flatten_associative(stripped.args[0], op) + _flatten_associative(
            stripped.args[1], op
        )
    return [expr]


def grouped_reduction_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[str, str] | None:
    """Classify a primitive or unrolled grouped reduction loop body."""
    allowed_conversion_dtypes = frozenset((source_dtype, torch.float32))
    candidates = [expr for expr in _walk(store.value) if expr.op == "reduction"]
    matches = []
    for reduction in candidates:
        reduction_type = str(reduction.args[2])
        source_type = _source_transform(
            reduction.args[3], source_name, allowed_conversion_dtypes
        )
        if source_type is not None:
            matches.append((reduction_type, source_type))
    if candidates:
        return matches[0] if len(candidates) == len(matches) == 1 else None

    root = _strip_conversions(store.value)
    while isinstance(root, GemmEpilogueIRExpression) and root.op in (
        "add",
        "mul",
        "truediv",
    ):
        if root.op == "truediv" and _constant_value(root.args[1]) == group:
            terms = _flatten_associative(root.args[0], "add")
            transforms = [
                _source_transform(term, source_name, allowed_conversion_dtypes)
                for term in terms
            ]
            if (
                len(terms) == group
                and len(frozenset(transforms)) == 1
                and transforms[0]
            ):
                return "mean", transforms[0]
        terms = _flatten_associative(root, root.op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return ("sum" if root.op == "add" else "prod"), transforms[0]
        break
    for op, reduction_type in (("maximum", "max"), ("minimum", "min")):
        terms = _flatten_associative(root, op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return reduction_type, transforms[0]
    return None


def _synthetic_reductions_ir(
    expr: Any,
    index: sympy.Expr,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[GemmEpilogueIRReduction, ...]:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return ()
    classified = grouped_reduction_ir(
        GemmEpilogueIRStore(index, expr), source_name, group, source_dtype
    )
    if classified is not None:
        reduction_type, source_type = classified
        return (GemmEpilogueIRReduction(reduction_type, expr, source_type=source_type),)
    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for arg in expr.args:
        for value in _expression_values(arg):
            for reduction in _synthetic_reductions_ir(
                value, index, source_name, group, source_dtype
            ):
                if id(reduction.source) not in seen:
                    seen.add(id(reduction.source))
                    reductions.append(reduction)
    return tuple(reductions)


def is_direct_bool_gt_zero_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Match a direct comparison of one lowered buffer load against zero."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op not in ("gt", "ge"):
        return False
    lhs, rhs = map(_strip_conversions, expr.args[:2])
    return (
        isinstance(lhs, GemmEpilogueIRExpression)
        and lhs.op == "load"
        and lhs.args[0] == source_name
        and _constant_value(rhs) == 0
    )


def is_absmax_scale_finalizer_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Match clamp(source, 1e-12) / 448 used by FP8 absmax scaling."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    if expr.op == "truediv" and _constant_value(expr.args[1]) == 448.0:
        clamped = expr.args[0]
    elif expr.op == "mul":
        lhs, rhs = expr.args[:2]
        if _constant_value(lhs) == 1.0 / 448.0:
            clamped = rhs
        elif _constant_value(rhs) == 1.0 / 448.0:
            clamped = lhs
        else:
            return False
    else:
        return False

    clamped = _strip_conversions(clamped)
    if not isinstance(clamped, GemmEpilogueIRExpression):
        return False
    if clamped.op in ("maximum", "clamp_min"):
        lhs, rhs = clamped.args[:2]
        return (
            _source_transform(lhs, source_name) == "identity"
            and _constant_value(rhs) == 1e-12
        ) or (
            _source_transform(rhs, source_name) == "identity"
            and _constant_value(lhs) == 1e-12
        )
    if clamped.op == "clamp" and len(clamped.args) >= 2:
        return (
            _source_transform(clamped.args[0], source_name) == "identity"
            and _constant_value(clamped.args[1]) == 1e-12
            and (len(clamped.args) < 3 or clamped.args[2] is None)
        )
    return False


def _contains_reduction(expr: Any) -> bool:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    return expr.op == "reduction" or any(_contains_reduction(arg) for arg in expr.args)


def _affine_scale(
    value: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return value[0] * scale, value[1] * scale, value[2] * scale


def _affine_add(
    lhs: tuple[float, float, float], rhs: tuple[float, float, float]
) -> tuple[float, float, float]:
    return lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]


def _affine_coefficients(
    expr: Any, source_name: str, reduction_names: frozenset[str]
) -> tuple[float, float, float] | None:
    expr = _strip_conversions(expr)
    if isinstance(expr, (int, float, sympy.Number)) and not isinstance(expr, bool):
        return 0.0, 0.0, float(expr)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        name, _, stored = expr.args
        if name == source_name:
            return 1.0, 0.0, 0.0
        if name in reduction_names or _contains_reduction(stored):
            return 0.0, 1.0, 0.0
        return _affine_coefficients(stored, source_name, reduction_names)
    if expr.op == "reduction":
        return 0.0, 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, 0.0, float(constant)
    if expr.op == "neg":
        value = _affine_coefficients(expr.args[0], source_name, reduction_names)
        return None if value is None else _affine_scale(value, -1.0)
    if expr.op in ("add", "sub"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        scale = 1.0 if expr.op == "add" else -1.0
        return _affine_add(lhs, _affine_scale(rhs, scale))
    if expr.op in ("mul", "truediv"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        if expr.op == "truediv":
            if rhs[:2] != (0.0, 0.0) or rhs[2] == 0.0:
                return None
            return _affine_scale(lhs, 1.0 / rhs[2])
        if lhs[:2] == (0.0, 0.0):
            return _affine_scale(rhs, lhs[2])
        if rhs[:2] == (0.0, 0.0):
            return _affine_scale(lhs, rhs[2])
    if expr.op == "fma":
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        addend = _affine_coefficients(expr.args[2], source_name, reduction_names)
        if lhs is None or rhs is None or addend is None:
            return None
        if lhs[:2] == (0.0, 0.0):
            product = _affine_scale(rhs, lhs[2])
        elif rhs[:2] == (0.0, 0.0):
            product = _affine_scale(lhs, rhs[2])
        else:
            return None
        return _affine_add(product, addend)
    return None


def centered_mean_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    reduction_scale: float = 1.0,
) -> str | None:
    """Classify an affine combination of a lowered value and grouped reduction."""
    coefficients = _affine_coefficients(store.value, source_name, reduction_names)
    if coefficients is not None:
        coefficients = (
            coefficients[0],
            coefficients[1] * reduction_scale,
            coefficients[2],
        )
    if (
        coefficients is None
        or coefficients[0] == 0.0
        or coefficients[1] == 0.0
        or not all(math.isfinite(value) for value in coefficients)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in coefficients)


def single_source_affine_ir(
    store: GemmEpilogueIRStore, source_name: str
) -> tuple[float, float] | None:
    """Return scale and bias for an affine expression of one lowered buffer."""
    coefficients = _affine_coefficients(store.value, source_name, frozenset())
    if coefficients is None or coefficients[1] != 0.0:
        return None
    return coefficients[0], coefficients[2]


def _affine_around(expr: Any, basis) -> tuple[float, float] | None:
    expr = _strip_conversions(expr)
    if basis(expr):
        return 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, float(constant)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "neg":
        value = _affine_around(expr.args[0], basis)
        return None if value is None else (-value[0], -value[1])
    if expr.op in ("add", "sub"):
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        sign = 1.0 if expr.op == "add" else -1.0
        return lhs[0] + sign * rhs[0], lhs[1] + sign * rhs[1]
    if expr.op == "mul":
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        if lhs[0] == 0.0:
            return rhs[0] * lhs[1], rhs[1] * lhs[1]
        if rhs[0] == 0.0:
            return lhs[0] * rhs[1], lhs[1] * rhs[1]
    return None


def _is_source(expr: Any, source_name: str) -> bool:
    return _source_transform(expr, source_name) == "identity"


def _sum_terms(expr: Any, source_name: str, group: int) -> bool:
    terms = _flatten_associative(expr, "add")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _group_max(expr: Any, source_name: str, group: int) -> bool:
    terms = _flatten_associative(expr, "maximum")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _stable_group_max(expr: Any, source_name: str, group: int) -> bool:
    """Match max(xs), including logsumexp's where(isinf(max), 0, max)."""
    expr = _strip_conversions(expr)
    if _group_max(expr, source_name, group):
        return True
    if not (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op == "where"
        and len(expr.args) >= 3
        and _constant_value(expr.args[1]) == 0.0
    ):
        return False
    condition = _strip_conversions(expr.args[0])
    maximum = _strip_conversions(expr.args[2])
    if not (
        isinstance(condition, GemmEpilogueIRExpression)
        and condition.op == "eq"
        and _group_max(maximum, source_name, group)
    ):
        return False
    for absolute, infinity in (condition.args[:2], reversed(condition.args[:2])):
        absolute = _strip_conversions(absolute)
        if (
            isinstance(absolute, GemmEpilogueIRExpression)
            and absolute.op == "abs"
            and _strip_conversions(absolute.args[0]) == maximum
            and _constant_value(infinity) == math.inf
        ):
            return True
    return False


def _shifted_exp(expr: Any, source_name: str) -> Any | None:
    expr = _strip_conversions(expr)
    if not (
        isinstance(expr, GemmEpilogueIRExpression) and expr.op == "exp" and expr.args
    ):
        return None
    shifted = _strip_conversions(expr.args[0])
    if not (
        isinstance(shifted, GemmEpilogueIRExpression)
        and shifted.op == "sub"
        and _is_source(shifted.args[0], source_name)
    ):
        return None
    return _strip_conversions(shifted.args[1])


def is_softmax_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    reduction_names: frozenset[str] = frozenset(),
) -> bool:
    """Match stable grouped softmax, either unrolled or reduction-backed."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
        return False
    numerator, denominator = expr.args[:2]
    maximum = _shifted_exp(numerator, source_name)
    if maximum is None:
        return False
    denominator = _strip_conversions(denominator)
    if reduction_names:
        return (
            isinstance(maximum, GemmEpilogueIRExpression)
            and maximum.op == "load"
            and maximum.args[0] in reduction_names
            and isinstance(denominator, GemmEpilogueIRExpression)
            and denominator.op == "load"
            and denominator.args[0] in reduction_names
            and maximum.args[0] != denominator.args[0]
        )
    terms = _flatten_associative(denominator, "add")
    return (
        _group_max(maximum, source_name, group)
        and len(terms) == group
        and all(_shifted_exp(term, source_name) == maximum for term in terms)
    )


def is_absmax_normalize_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    scale_names: str | frozenset[str],
) -> bool:
    """Match source * reciprocal(clamp(absmax, 1e-12) / 448)."""
    if isinstance(scale_names, str):
        scale_names = frozenset((scale_names,))
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return False
    for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
        reciprocal = _strip_conversions(reciprocal)
        if (
            _is_source(source, source_name)
            and isinstance(reciprocal, GemmEpilogueIRExpression)
            and reciprocal.op == "reciprocal"
            and any(
                is_absmax_scale_finalizer_ir(
                    GemmEpilogueIRStore(store.index, reciprocal.args[0]), scale_name
                )
                for scale_name in scale_names
            )
        ):
            return True
    return False


def _sum_affine_ir(
    expr: Any,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> tuple[float, float] | None:
    def is_sum(candidate: Any) -> bool:
        candidate = _strip_conversions(candidate)
        return (
            isinstance(candidate, GemmEpilogueIRExpression)
            and candidate.op == "load"
            and candidate.args[0] in reduction_names
        ) or _sum_terms(candidate, source_name, group)

    return _affine_around(expr, is_sum)


def sum_normalize_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    """Classify normalization by a primitive or unrolled grouped sum."""
    parameters: tuple[str, float, float] | None = None

    def is_normalization(expr: Any) -> bool:
        nonlocal parameters
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return False
        if expr.op == "truediv":
            lhs, rhs = expr.args[:2]
            if _is_source(lhs, source_name):
                affine = _sum_affine_ir(rhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "forward", *affine
                    return True
            if _is_source(rhs, source_name):
                affine = _sum_affine_ir(lhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "reverse", *affine
                    return True
        if expr.op == "mul":
            for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
                reciprocal = _strip_conversions(reciprocal)
                if (
                    _is_source(source, source_name)
                    and isinstance(reciprocal, GemmEpilogueIRExpression)
                    and reciprocal.op == "reciprocal"
                ):
                    affine = _sum_affine_ir(
                        reciprocal.args[0], source_name, reduction_names, group
                    )
                    if affine is not None:
                        parameters = "forward", *affine
                        return True
        return False

    affine = _affine_around(store.value, is_normalization)
    if (
        affine is None
        or parameters is None
        or affine[0] == 0.0
        or not all(math.isfinite(value) for value in (*affine, *parameters[1:]))
    ):
        return None
    kind = (
        "normalize_sum_affine"
        if parameters[0] == "forward"
        else "normalize_sum_reverse_affine"
    )
    values = (*affine, *parameters[1:])
    return kind + ":" + ":".join(format(value, ".17g") for value in values)


def sum_multiply_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return None
    for source, reduction in (expr.args[:2], reversed(expr.args[:2])):
        if not _is_source(source, source_name):
            continue
        affine = _sum_affine_ir(reduction, source_name, reduction_names, group)
        if affine is not None and all(math.isfinite(value) for value in affine):
            return "sum_mul_affine:" + ":".join(
                format(value, ".17g") for value in affine
            )
    return None


def variance_parameters_ir(
    store: GemmEpilogueIRStore, source_name: str, group: int
) -> tuple[float, float] | None:
    """Match an unrolled grouped variance followed by an affine transform."""

    def is_variance(expr: Any) -> bool:
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
            return False
        if _constant_value(expr.args[1]) != group:
            return False
        squares = _flatten_associative(expr.args[0], "add")
        if len(squares) != group:
            return False
        for square in squares:
            square = _strip_conversions(square)
            if not (
                isinstance(square, GemmEpilogueIRExpression)
                and square.op == "mul"
                and square.args[0] == square.args[1]
            ):
                return False
            centered = _strip_conversions(square.args[0])
            if not (
                isinstance(centered, GemmEpilogueIRExpression)
                and centered.op == "sub"
                and _is_source(centered.args[0], source_name)
            ):
                return False
            mean = _strip_conversions(centered.args[1])
            if not (
                isinstance(mean, GemmEpilogueIRExpression)
                and mean.op == "truediv"
                and _constant_value(mean.args[1]) == group
                and _sum_terms(mean.args[0], source_name, group)
            ):
                return False
        return True

    affine = _affine_around(store.value, is_variance)
    return affine if affine is not None and affine[0] != 0.0 else None


def centered_mean_consumer_type_unrolled_ir(
    store: GemmEpilogueIRStore, source_name: str, group: int
) -> str | None:
    """Classify an affine source/mean expression after a small reduction unroll."""

    def coefficients(expr: Any) -> tuple[float, float, float] | None:
        expr = _strip_conversions(expr)
        if isinstance(expr, GemmEpilogueIRExpression) and expr.op == "truediv":
            if _constant_value(expr.args[1]) == group and _sum_terms(
                expr.args[0], source_name, group
            ):
                return 0.0, 1.0, 0.0
        if _is_source(expr, source_name):
            return 1.0, 0.0, 0.0
        constant = _constant_value(expr)
        if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
            constant, bool
        ):
            return 0.0, 0.0, float(constant)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return None
        if expr.op in ("add", "sub"):
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            return _affine_add(
                lhs, _affine_scale(rhs, 1.0 if expr.op == "add" else -1.0)
            )
        if expr.op == "mul":
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            if lhs[:2] == (0.0, 0.0):
                return _affine_scale(rhs, lhs[2])
            if rhs[:2] == (0.0, 0.0):
                return _affine_scale(lhs, rhs[2])
        return None

    values = coefficients(store.value)
    if (
        values is None
        or values[0] == 0.0
        or values[1] == 0.0
        or not all(math.isfinite(value) for value in values)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in values)


def is_logsumexp_ir(store: GemmEpilogueIRStore, source_name: str, group: int) -> bool:
    """Match max(xs) + log(sum(exp(xs - max(xs))))."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "add":
        return False
    for maximum, logarithm in (expr.args[:2], reversed(expr.args[:2])):
        maximum = _strip_conversions(maximum)
        logarithm = _strip_conversions(logarithm)
        if not (
            _stable_group_max(maximum, source_name, group)
            and isinstance(logarithm, GemmEpilogueIRExpression)
            and logarithm.op == "log"
        ):
            continue
        terms = _flatten_associative(logarithm.args[0], "add")
        if len(terms) == group and all(
            _shifted_exp(term, source_name) == maximum for term in terms
        ):
            return True
    return False
