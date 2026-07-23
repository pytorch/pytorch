# mypy: allow-untyped-defs
"""Backend-neutral FX expression helpers for GEMM epilogues."""

import dataclasses
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    statically_known_true as fx_statically_known_true,
)
from torch.utils._ordered_set import OrderedSet


def statically_known(expr: Any) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )


@dataclasses.dataclass(frozen=True)
class GemmReductionGeometry:
    """Describe the grouped output axis shared by GEMM reduction consumers."""

    group: int
    axis: int

    def __post_init__(self) -> None:
        if self.group <= 0:
            raise RuntimeError("local_reduce_group must be positive")
        if self.axis not in (0, 1):
            raise RuntimeError("local_reduce_axis must be 0 or 1")

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.axis == 0 or self.group > 32


@dataclasses.dataclass(frozen=True)
class GroupedReductionLayout:
    """Describe a grouped view over one axis of a two-dimensional GEMM output."""

    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    def matches_reduction_dim(self, dim: Any) -> bool:
        dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        return len(dims) == 1 and dims[0] in self.reduce_dims

    def matches_output_shape(
        self, output_shape: Sequence[Any], gemm_shape: Sequence[Any]
    ) -> bool:
        if len(gemm_shape) != 2:
            return False
        m, n = gemm_shape
        grouped = (
            (m, n // self.group_size, self.group_size)
            if self.axis == 1
            else (m // self.group_size, self.group_size, n)
        )
        return statically_known_shape_equal(
            output_shape, (m, n)
        ) or statically_known_shape_equal(output_shape, grouped)


def iter_fx_node_inputs(value: Any) -> Iterator[torch.fx.Node]:
    """Yield FX node inputs nested in args/kwargs-style containers."""
    result: list[torch.fx.Node] = []
    torch.fx.map_arg(value, lambda node: result.append(node))
    yield from result


@dataclasses.dataclass(frozen=True)
class GemmEpilogueGraph:
    """Index transitive dependencies between nodes in an epilogue FX graph."""

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]

    @classmethod
    def from_nodes(cls, nodes: Sequence[torch.fx.Node]) -> "GemmEpilogueGraph":
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        for node in nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
        return cls(dependencies)

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "GemmEpilogueGraph":
        return cls.from_nodes(tuple(graph_module.graph.nodes))

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )


def epilogue_subgraph_from_origins(
    origins: Sequence[torch.fx.Node],
) -> tuple[tuple[torch.fx.Node, ...], tuple[torch.fx.Node, ...]]:
    """Return the ordered ancestor subgraph and roots for scheduler origins."""
    origin_set = OrderedSet(origins)
    if not origin_set:
        raise RuntimeError("GEMM epilogue analysis requires FX origin nodes")
    graphs = OrderedSet(node.graph for node in origin_set)
    if len(graphs) != 1:
        raise RuntimeError("GEMM epilogue origins must belong to one FX graph")
    graph = next(iter(graphs))
    ancestors = origin_set.copy()
    pending = list(origin_set)
    while pending:
        node = pending.pop()
        for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
            if input_node not in ancestors:
                ancestors.add(input_node)
                pending.append(input_node)
    nodes = tuple(node for node in graph.nodes if node in ancestors)
    roots = tuple(
        node
        for node in origin_set
        if not any(user in origin_set for user in node.users)
    )
    return nodes, roots


def fx_target(value: Any) -> str:
    return str(getattr(value, "target", ""))


def fx_targets(values: Iterable[Any]) -> OrderedSet[str]:
    """Collect non-empty FX target names while preserving graph order."""
    return OrderedSet(target for value in values if (target := fx_target(value)))


def fx_contains(value: Any, target: str) -> bool:
    if fx_target(value) == target:
        return True
    return any(
        fx_contains(arg, target)
        for arg in getattr(value, "args", ())
        if hasattr(arg, "target")
    )


def fx_expression_roots(
    origins: Collection[Any], targets: Collection[str]
) -> list[Any]:
    expressions = [origin for origin in origins if fx_target(origin) in targets]
    referenced = OrderedSet(
        arg
        for expression in expressions
        for arg in expression.args
        if hasattr(arg, "target") and arg in origins
    )
    return [expression for expression in expressions if expression not in referenced]


def lower_affine_expression(
    value: Any,
    basis: Callable[[Any], int | None],
    basis_count: int,
    *,
    wrappers: Collection[str] = (),
) -> tuple[float, ...] | None:
    """Lower an FX add/sub/mul expression to basis coefficients and a constant."""
    if isinstance(value, (int, float)):
        return (0.0,) * basis_count + (float(value),)
    basis_index = basis(value)
    if basis_index is not None:
        if not 0 <= basis_index < basis_count:
            raise AssertionError("epilogue affine basis index is out of range")
        return tuple(
            1.0 if index == basis_index else 0.0 for index in range(basis_count)
        ) + (0.0,)
    target = fx_target(value)
    if target in wrappers:
        args = getattr(value, "args", ())
        return (
            lower_affine_expression(args[0], basis, basis_count, wrappers=wrappers)
            if args
            else None
        )
    if target in ("aten.add.Tensor", "aten.sub.Tensor"):
        lhs = lower_affine_expression(
            value.args[0], basis, basis_count, wrappers=wrappers
        )
        rhs = lower_affine_expression(
            value.args[1], basis, basis_count, wrappers=wrappers
        )
        if lhs is None or rhs is None:
            return None
        alpha = float(value.kwargs.get("alpha", 1.0))
        rhs_scale = alpha if target == "aten.add.Tensor" else -alpha
        return tuple(a + rhs_scale * b for a, b in zip(lhs, rhs))
    if target == "aten.mul.Tensor":
        lhs = lower_affine_expression(
            value.args[0], basis, basis_count, wrappers=wrappers
        )
        rhs = lower_affine_expression(
            value.args[1], basis, basis_count, wrappers=wrappers
        )
        if lhs is None or rhs is None:
            return None
        if all(coefficient == 0.0 for coefficient in lhs[:-1]):
            return tuple(coefficient * lhs[-1] for coefficient in rhs)
        if all(coefficient == 0.0 for coefficient in rhs[:-1]):
            return tuple(coefficient * rhs[-1] for coefficient in lhs)
    return None


_SUM_AFFINE_WRAPPERS = (
    "aten.clone.default",
    "aten.expand.default",
    "aten.reshape.default",
    "aten.view.default",
    "aten._unsafe_view.default",
)


def _sum_affine(value: Any) -> tuple[float, ...] | None:
    return lower_affine_expression(
        value,
        lambda candidate: (
            0 if fx_target(candidate) == "aten.sum.dim_IntList" else None
        ),
        1,
        wrappers=_SUM_AFFINE_WRAPPERS,
    )


def centered_mean_consumer_type(origins: Collection[Any]) -> str | None:
    """Classify an affine combination of a value and its grouped mean."""
    wrappers = (*_SUM_AFFINE_WRAPPERS, "prims.convert_element_type.default")

    def basis(value: Any) -> int | None:
        target = fx_target(value)
        if target == "aten.mean.dim":
            return 1
        if target == "prims.convert_element_type.default" and not fx_contains(
            value, "aten.mean.dim"
        ):
            return 0
        return None

    roots = fx_expression_roots(
        origins, ("aten.add.Tensor", "aten.mul.Tensor", "aten.sub.Tensor")
    )
    for root in roots:
        coefficients = lower_affine_expression(root, basis, 2, wrappers=wrappers)
        if (
            coefficients is not None
            and coefficients[0] != 0.0
            and coefficients[1] != 0.0
            and all(math.isfinite(value) for value in coefficients)
        ):
            return "mean_linear:" + ":".join(
                format(value, ".17g") for value in coefficients
            )
    return None


def sum_normalize_consumer_type(origins: Collection[Any]) -> str | None:
    """Classify an affine expression containing input/sum normalization."""

    def normalize_kind(value: Any) -> tuple[str, float, float] | None:
        target = fx_target(value)
        if target == "aten.mul.Tensor":
            lhs, rhs = value.args[:2]
            reciprocal = None
            if fx_contains(lhs, "aten.reciprocal.default") and fx_contains(
                rhs, "prims.convert_element_type.default"
            ):
                reciprocal = lhs
            elif fx_contains(rhs, "aten.reciprocal.default") and fx_contains(
                lhs, "prims.convert_element_type.default"
            ):
                reciprocal = rhs
            if reciprocal is not None:
                while fx_target(reciprocal) != "aten.reciprocal.default":
                    reciprocal = reciprocal.args[0]
                affine = _sum_affine(reciprocal.args[0])
                if affine is not None:
                    return "forward", affine[0], affine[1]
        if target == "aten.div.Tensor":
            lhs, rhs = value.args[:2]
            lhs_sum = fx_contains(lhs, "aten.sum.dim_IntList")
            rhs_sum = fx_contains(rhs, "aten.sum.dim_IntList")
            lhs_input = fx_contains(lhs, "prims.convert_element_type.default")
            rhs_input = fx_contains(rhs, "prims.convert_element_type.default")
            if lhs_input and rhs_sum:
                affine = _sum_affine(rhs)
                if affine is not None:
                    return "forward", affine[0], affine[1]
            if lhs_sum and rhs_input:
                affine = _sum_affine(lhs)
                if affine is not None:
                    return "reverse", affine[0], affine[1]
        return None

    roots = fx_expression_roots(
        origins,
        ("aten.add.Tensor", "aten.div.Tensor", "aten.mul.Tensor", "aten.sub.Tensor"),
    )
    normalization_parameters = None

    def normalization_basis(value: Any) -> int | None:
        nonlocal normalization_parameters
        kind = normalize_kind(value)
        if kind is None:
            return None
        normalization_parameters = kind[1:]
        return 0 if kind[0] == "forward" else 1

    for root in roots:
        coefficients = lower_affine_expression(
            root,
            normalization_basis,
            2,
            wrappers=(
                "aten.clone.default",
                "aten.reshape.default",
                "aten.view.default",
                "aten._unsafe_view.default",
            ),
        )
        if (
            coefficients is not None
            and normalization_parameters is not None
            and (coefficients[0] != 0.0) != (coefficients[1] != 0.0)
            and all(math.isfinite(value) for value in coefficients)
        ):
            kind = (
                "normalize_sum_affine"
                if coefficients[0]
                else "normalize_sum_reverse_affine"
            )
            values = (
                coefficients[0] or coefficients[1],
                coefficients[2],
                *normalization_parameters,
            )
            return kind + ":" + ":".join(format(value, ".17g") for value in values)
    return None


def sum_multiply_consumer_type(origins: Collection[Any]) -> str | None:
    """Classify multiplication of an input by an affine grouped sum."""
    for origin in origins:
        if fx_target(origin) != "aten.mul.Tensor":
            continue
        lhs, rhs = origin.args[:2]
        for input_value, reduction_value in ((lhs, rhs), (rhs, lhs)):
            if not fx_contains(
                input_value, "prims.convert_element_type.default"
            ) or not fx_contains(reduction_value, "aten.sum.dim_IntList"):
                continue
            affine = _sum_affine(reduction_value)
            if affine is not None and all(math.isfinite(value) for value in affine):
                return "sum_mul_affine:" + ":".join(
                    format(value, ".17g") for value in affine
                )
    return None


def _tensor_and_scalar(value: Any, target: str) -> tuple[Any, float] | None:
    if fx_target(value) != target:
        return None
    args = getattr(value, "args", ())
    if len(args) < 2:
        return None
    if isinstance(args[0], (int, float)):
        return args[1], float(args[0])
    if isinstance(args[1], (int, float)):
        return args[0], float(args[1])
    return None


def grouped_variance_parameters(origins: Collection[Any]) -> tuple[float, float] | None:
    """Match affine grouped variance and return its scale and bias."""
    allowed = frozenset(
        (
            "aten.add.Tensor",
            "aten.mean.dim",
            "aten.mul.Tensor",
            "aten.pow.Tensor_Scalar",
            "aten.reshape.default",
            "aten.sub.Tensor",
            "prims.convert_element_type.default",
        )
    )
    targets = fx_targets(origins)
    if not targets or not targets.issubset(allowed):
        return None
    matches: list[tuple[float, float]] = []
    for add in origins:
        add_parts = _tensor_and_scalar(add, "aten.add.Tensor")
        if add_parts is None:
            continue
        mul, bias = add_parts
        mul_parts = _tensor_and_scalar(mul, "aten.mul.Tensor")
        if mul_parts is None:
            continue
        mean, scale = mul_parts
        mean_args = getattr(mean, "args", ())
        if fx_target(mean) != "aten.mean.dim" or (
            len(mean_args) < 2 or tuple(mean_args[1]) != (-1,)
        ):
            continue
        square_parts = _tensor_and_scalar(mean_args[0], "aten.pow.Tensor_Scalar")
        if square_parts is None or square_parts[1] != 2.0:
            continue
        centered_args = getattr(square_parts[0], "args", ())
        if fx_target(square_parts[0]) != "aten.sub.Tensor" or len(centered_args) < 2:
            continue
        grouped, group_mean = centered_args[:2]
        group_mean_args = getattr(group_mean, "args", ())
        if (
            fx_target(group_mean) != "aten.mean.dim"
            or len(group_mean_args) < 3
            or group_mean_args[0] is not grouped
            or tuple(group_mean_args[1]) != (-1,)
            or group_mean_args[2] is not True
        ):
            continue
        matches.append((scale, bias))
    return matches[0] if len(matches) == 1 else None


def _logsumexp_shift_matches(grouped: Any, shift: Any, guarded: bool) -> bool:
    if guarded:
        where_args = getattr(shift, "args", ())
        if fx_target(shift) != "aten.where.self" or len(where_args) < 3:
            return False
        condition, zero, maximum = where_args[:3]
        condition_args = getattr(condition, "args", ())
        zero_args = getattr(zero, "args", ())
        if (
            fx_target(condition) != "aten.eq.Scalar"
            or len(condition_args) < 2
            or fx_target(condition_args[0]) != "aten.abs.default"
            or getattr(condition_args[0], "args", (None,))[0] is not maximum
            or condition_args[1] != float("inf")
            or fx_target(zero) != "aten.full.default"
            or len(zero_args) < 2
            or zero_args[1] != 0.0
        ):
            return False
    else:
        maximum = shift
    maximum_args = getattr(maximum, "args", ())
    return bool(
        fx_target(maximum) == "aten.amax.default"
        and len(maximum_args) >= 3
        and maximum_args[0] is grouped
        and tuple(maximum_args[1]) == (-1,)
        and maximum_args[2] is True
    )


def is_grouped_logsumexp(origins: Collection[Any]) -> bool:
    """Match stable or nonfinite-guarded grouped logsumexp."""
    guarded_required = frozenset(
        (
            "aten.abs.default",
            "aten.add.Tensor",
            "aten.amax.default",
            "aten.eq.Scalar",
            "aten.exp.default",
            "aten.full.default",
            "aten.log.default",
            "aten.reshape.default",
            "aten.squeeze.dims",
            "aten.sub.Tensor",
            "aten.sum.dim_IntList",
            "aten.where.self",
            "prims.convert_element_type.default",
        )
    )
    stable_required = frozenset(
        (
            "aten.add.Tensor",
            "aten.amax.default",
            "aten.exp.default",
            "aten.log.default",
            "aten.reshape.default",
            "aten.squeeze.dim",
            "aten.sub.Tensor",
            "aten.sum.dim_IntList",
            "prims.convert_element_type.default",
        )
    )
    targets = fx_targets(origins)
    if targets == guarded_required:
        guarded = True
        squeeze_target = "aten.squeeze.dims"
    elif targets == stable_required:
        guarded = False
        squeeze_target = "aten.squeeze.dim"
    else:
        return False
    for add in origins:
        if fx_target(add) != "aten.add.Tensor" or len(add.args) < 2:
            continue
        log_node, squeeze = add.args[:2]
        if fx_target(log_node) != "aten.log.default":
            log_node, squeeze = squeeze, log_node
        squeeze_args = getattr(squeeze, "args", ())
        if (
            fx_target(log_node) != "aten.log.default"
            or fx_target(squeeze) != squeeze_target
            or len(squeeze_args) < 2
            or (tuple(squeeze_args[1]) if guarded else squeeze_args[1])
            != ((-1,) if guarded else -1)
        ):
            continue
        sum_node = getattr(log_node, "args", (None,))[0]
        sum_args = getattr(sum_node, "args", ())
        if (
            fx_target(sum_node) != "aten.sum.dim_IntList"
            or len(sum_args) < 2
            or tuple(sum_args[1]) != (-1,)
        ):
            continue
        exp = sum_args[0]
        sub = getattr(exp, "args", (None,))[0]
        sub_args = getattr(sub, "args", ())
        if (
            fx_target(exp) != "aten.exp.default"
            or fx_target(sub) != "aten.sub.Tensor"
            or len(sub_args) < 2
        ):
            continue
        grouped, shift = sub_args[:2]
        if squeeze_args[0] is shift and _logsumexp_shift_matches(
            grouped, shift, guarded
        ):
            return True
    return False
