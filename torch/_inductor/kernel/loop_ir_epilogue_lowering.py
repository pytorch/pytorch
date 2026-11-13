# mypy: allow-untyped-defs
"""Lower GEMM epilogue loop IR to shared epilogue contracts."""

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import sympy

import torch
from torch._inductor.ir import ComputedBuffer
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
    GemmReductionType,
)
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
    synthetic_element: GemmEpilogueIRExpression | None = None


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRRegion:
    reductions: tuple[GemmEpilogueIRReduction, ...]
    expression: GemmEpilogueIRExpression


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRSyntheticReduction:
    geometry: GemmReductionGeometry
    region: GemmEpilogueIRRegion


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
class GemmEpilogueIRFinalizer:
    """Normalized operation applied to a completed grouped reduction."""

    output_name: str
    source_name: str
    materialize: bool


def _expression_values(value: object):
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
                        GemmEpilogueIRReduction(reduction_type, value),
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


def _walk(expr: object, *, follow_stored_values: bool = True):
    if not isinstance(expr, GemmEpilogueIRExpression):
        return
    yield expr
    args = (
        expr.args[:2] if expr.op == "load" and not follow_stored_values else expr.args
    )
    for arg in args:
        yield from _walk(arg, follow_stored_values=follow_stored_values)


def _loaded_names(expr: object, *, follow_stored_values: bool = True) -> frozenset[str]:
    return frozenset(
        value.args[0]
        for value in _walk(expr, follow_stored_values=follow_stored_values)
        if value.op == "load"
    )


def _contains_reduction(expr: object, *, follow_stored_values: bool = True) -> bool:
    return any(
        value.op == "reduction"
        for value in _walk(expr, follow_stored_values=follow_stored_values)
    )


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRAnalysis:
    """Candidate-wide semantic view of existing lowered loop bodies."""

    stores: dict[str, GemmEpilogueIRStore]
    buffers: tuple[ComputedBuffer, ...] = ()
    index_vars: dict[str, tuple[sympy.Expr, ...]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_buffers(
        cls, buffers: Sequence[ComputedBuffer]
    ) -> "GemmEpilogueIRAnalysis":
        buffers = tuple(buffers)
        handler = _GemmEpilogueIRHandler()
        index_vars = {}
        with V.set_ops_handler(handler):
            for buffer in buffers:
                args = buffer.data.inner_fn_args()
                if len(args) == 1:
                    index_vars[buffer.get_name()] = tuple(args[0])
                buffer.get_store_function()(*args)
        return cls(handler.stores, buffers, index_vars)

    def store(self, name: str) -> GemmEpilogueIRStore | None:
        return self.stores.get(name)

    def _source_load_width(self, output_name: str, source_name: str) -> int:
        store = self.store(output_name)
        if store is None:
            return 0
        indices = [
            expr.args[1]
            for expr in _walk(store.value)
            if expr.op == "load" and expr.args[0] == source_name
        ]
        unique_indices = []
        for index in indices:
            if not any(sympy.simplify(index - other) == 0 for other in unique_indices):
                unique_indices.append(index)
        return len(unique_indices)

    def synthetic_reduction_region(
        self,
        output_name: str,
        source_name: str,
        source_dtype: torch.dtype,
        n: int,
    ) -> GemmEpilogueIRSyntheticReduction | None:
        match = self.synthetic_reduction_program(
            output_name, source_name, source_dtype, n
        )
        if match is None or len(match.region.reductions) != 1:
            return None
        return match

    def synthetic_reduction_program(
        self,
        output_name: str,
        source_name: str,
        source_dtype: torch.dtype,
        n: int,
    ) -> GemmEpilogueIRSyntheticReduction | None:
        """Infer one grouped geometry shared by all unrolled reductions."""
        matches = []
        width = self._source_load_width(output_name, source_name)
        for group in range(2, width + 1):
            region = self.reduction_region(
                output_name, source_name, group, source_dtype
            )
            if region is None:
                continue
            axes = tuple(
                grouped_reduction_axis_ir(reduction, group, n)
                for reduction in region.reductions
            )
            if axes and axes[0] is not None and all(axis == axes[0] for axis in axes):
                matches.append(
                    GemmEpilogueIRSyntheticReduction(
                        GemmReductionGeometry(group, axes[0]), region
                    )
                )
        return max(matches, key=lambda match: match.geometry.group, default=None)

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
        return GemmEpilogueIRRegion(reductions, store.value)

    def reduction_finalizer(
        self,
        output_name: str,
        source_name: str,
    ) -> GemmEpilogueIRFinalizer | None:
        store = self.store(output_name)
        if store is None:
            return None
        direct_inputs = _loaded_names(store.value, follow_stored_values=False)
        has_direct_reduction = _contains_reduction(
            store.value, follow_stored_values=False
        )
        if operation_names_ir(store).issubset(
            ("load", "to_dtype", "to_dtype_bitcast", "identity")
        ):
            materialize = False
        elif direct_inputs == frozenset((source_name,)) and not has_direct_reduction:
            materialize = True
        else:
            return None
        return GemmEpilogueIRFinalizer(output_name, source_name, materialize)


def _constant_value(expr: object) -> Any | None:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op in ("constant", "index_expr") and expr.args:
        return expr.args[0]
    if expr.op in ("to_dtype", "to_dtype_bitcast") and expr.args:
        return _constant_value(expr.args[0])
    return None


def _strip_conversions(expr: object) -> Any:
    while (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op in ("to_dtype", "to_dtype_bitcast", "identity")
        and expr.args
    ):
        expr = expr.args[0]
    return expr


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


def _supports_reduction_source_conversions(
    expr: GemmEpilogueIRExpression, source_dtype: torch.dtype
) -> bool:
    allowed_dtypes = (source_dtype, torch.float32)
    for value in _walk(expr):
        if value.op == "to_dtype_bitcast":
            return False
        if value.op == "to_dtype" and (
            len(value.args) < 2 or value.args[1] not in allowed_dtypes
        ):
            return False
    return True


def _flatten_associative(expr: object, op: str) -> list[Any]:
    stripped = _strip_conversions(expr)
    if isinstance(stripped, GemmEpilogueIRExpression) and stripped.op == op:
        return _flatten_associative(stripped.args[0], op) + _flatten_associative(
            stripped.args[1], op
        )
    return [expr]


def _expression_pattern(expr: object, source_name: str) -> Any:
    """Canonicalize a pointwise expression while ignoring source load indices."""
    if isinstance(expr, GemmEpilogueIRExpression):
        if expr.op == "load" and expr.args[0] == source_name:
            return ("load", source_name)
        return (
            expr.op,
            tuple(_expression_pattern(arg, source_name) for arg in expr.args),
            tuple(
                (key, _expression_pattern(value, source_name))
                for key, value in expr.kwargs
            ),
        )
    if isinstance(expr, (tuple, list)):
        return tuple(_expression_pattern(item, source_name) for item in expr)
    return expr


def _synthetic_reduction_element_ir(
    expr: object,
    source_name: str,
    group: int,
) -> tuple[GemmReductionType, GemmEpilogueIRExpression] | None:
    root = _strip_conversions(expr)
    reduction_type: GemmReductionType
    if (
        isinstance(root, GemmEpilogueIRExpression)
        and root.op == "truediv"
        and _constant_value(root.args[1]) == group
    ):
        reduction_type = "mean"
        root = _strip_conversions(root.args[0])
        associative_op = "add"
    elif isinstance(root, GemmEpilogueIRExpression) and root.op in (
        "add",
        "mul",
        "maximum",
        "minimum",
    ):
        reduction_type = cast(
            GemmReductionType,
            {
                "add": "sum",
                "mul": "prod",
                "maximum": "max",
                "minimum": "min",
            }[root.op],
        )
        associative_op = root.op
    else:
        return None

    terms = _flatten_associative(root, associative_op)
    if len(terms) != group or not all(
        isinstance(term, GemmEpilogueIRExpression) for term in terms
    ):
        return None
    patterns = tuple(_expression_pattern(term, source_name) for term in terms)
    if any(pattern != patterns[0] for pattern in patterns[1:]):
        return None
    return reduction_type, cast(GemmEpilogueIRExpression, terms[0])


def grouped_reduction_pattern_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[GemmReductionType, GemmEpilogueIRExpression] | None:
    """Return the primitive reduction and its per-element source expression."""
    candidates = [expr for expr in _walk(store.value) if expr.op == "reduction"]
    matches = []
    for reduction in candidates:
        reduction_type = str(reduction.args[2])
        source = reduction.args[3]
        if (
            reduction_type in ("sum", "mean", "prod", "max", "min")
            and isinstance(source, GemmEpilogueIRExpression)
            and (source.loads or _loaded_names(source)) == frozenset((source_name,))
            and _supports_reduction_source_conversions(source, source_dtype)
        ):
            matches.append((cast(GemmReductionType, reduction_type), source))
    if candidates:
        return matches[0] if len(candidates) == len(matches) == 1 else None

    reductions = _synthetic_reductions_ir(
        store.value, store.index, source_name, group, source_dtype
    )
    if len(reductions) != 1:
        return None
    reduction = reductions[0]
    element = reduction.synthetic_element
    if element is None or not _supports_reduction_source_conversions(
        element, source_dtype
    ):
        return None
    return cast(GemmReductionType, reduction.reduction_type), element


def _synthetic_reductions_ir(
    expr: object,
    index: sympy.Expr,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[GemmEpilogueIRReduction, ...]:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return ()
    synthetic = _synthetic_reduction_element_ir(expr, source_name, group)
    if synthetic is not None:
        reduction_type, element = synthetic
        nested = _synthetic_reductions_ir(
            element, index, source_name, group, source_dtype
        )
        return (
            *nested,
            GemmEpilogueIRReduction(
                reduction_type,
                expr,
                synthetic_element=element,
            ),
        )
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
