# mypy: allow-untyped-defs
"""Grouped GEMM view and main-output layout recognition.

Local-reduction analysis and main-output planning share these layout rules;
reduction dataflow and kernel scheduling remain with their respective owners.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    FlexGemmStructuralInt,
    is_shape_preserving_pointwise_node,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    normalize_shape,
    statically_known_equal,
    statically_known_shape_equal,
)
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._inductor.kernel.gemm_epilogue_analysis import GemmLocalReduceAnalysis


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GemmReductionGeometry | None:
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    last = FlexGemmStructuralInt.from_value(shape[-1])
    penultimate = FlexGemmStructuralInt.from_value(shape[-2])
    if (
        last is not None
        and last.value > 0
        and type(shape[-2]) is int
        and shape[-2] == -1
    ):
        return GemmReductionGeometry(group=last.value, axis=1)
    if (
        type(shape[-3]) is int
        and shape[-3] == -1
        and penultimate is not None
        and penultimate.value > 0
    ):
        return GemmReductionGeometry(group=penultimate.value, axis=0)
    return None


def _group_count_matches_selected_dim(
    group_count: Any, selected_size: Any, group: int
) -> bool:
    if type(group_count) is int and group_count == -1:
        return True
    return statically_known_equal(
        group_count * group, selected_size
    ) or statically_known_equal(group_count, selected_size // group)


def _grouped_layout_matches_source_shape(
    shape: tuple[Any, ...],
    source_shape: tuple[Any, ...],
    layout: GemmReductionGeometry,
) -> bool:
    """Require a 2-D GEMM output reshape to split exactly M or N."""
    if len(shape) != 3:
        return False

    m, n = source_shape
    match layout.axis, shape:
        case 1, (kept_m, group_count, group):
            structural_group = FlexGemmStructuralInt.from_value(group)
            return (
                structural_group is not None
                and structural_group.value == layout.group
                and statically_known_equal(kept_m, m)
                and _group_count_matches_selected_dim(group_count, n, layout.group)
            )
        case 0, (group_count, group, kept_n):
            structural_group = FlexGemmStructuralInt.from_value(group)
            return (
                structural_group is not None
                and structural_group.value == layout.group
                and statically_known_equal(kept_n, n)
                and _group_count_matches_selected_dim(group_count, m, layout.group)
            )
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GemmReductionGeometry | None:
    """Recognize grouped M/N geometry, specializing backed group dimensions."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = tuple(shape[0])
    if source_shape is not None:
        source_shape = normalize_shape(source_shape)
        if isinstance(source_shape, tuple) and len(source_shape) == 2:
            candidates = []
            if shape:
                group = FlexGemmStructuralInt.from_value(shape[-1])
                if group is not None and group.value > 0:
                    candidates.append(GemmReductionGeometry(group=group.value, axis=1))
            if len(shape) >= 2:
                group = FlexGemmStructuralInt.from_value(shape[-2])
                if group is not None and group.value > 0:
                    candidates.append(GemmReductionGeometry(group=group.value, axis=0))
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


@dataclasses.dataclass(frozen=True)
class OutputContractionUse:
    """Describe one grouped-main lane before complete-output validation."""

    source: torch.fx.Node
    group: int
    chunked: bool
    index: int
    layout_node: torch.fx.Node
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()


@dataclasses.dataclass(frozen=True)
class OutputContractionPlan:
    """Describe one complete grouped-main output and its lowering metadata."""

    transform: FlexGemmOutputContraction
    select_indices: dict[torch.fx.Node, int]
    layouts: dict[torch.fx.Node, GemmReductionGeometry]
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()

    def commit_guards(self) -> None:
        """Specialize backed structural values after complete validation."""
        for structural in self.structural_values:
            structural.guard()


def canonical_output_contraction_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> torch.fx.Node:
    """Strip shape-preserving pointwise wrappers from a grouped lane source."""
    while node is not gemm and is_shape_preserving_pointwise_node(node):
        inputs = [
            arg
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if local_reduce.graph.depends_on(arg, gemm)
        ]
        if len(inputs) != 1:
            break
        node = inputs[0]
    return node


def match_output_contraction_use(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionUse | None:
    """Match one interleaved select or contiguous split lane."""
    normalized = local_reduce.graph.normalized_nodes.get(node)
    if isinstance(normalized, NormalizedSelect):
        view = normalized.source
        view_normalized = local_reduce.graph.normalized_nodes.get(view)
        dim = FlexGemmStructuralInt.from_value(normalized.dim)
        index = FlexGemmStructuralInt.from_value(normalized.index)
        shape = tensor_meta_shape(view)
        if (
            not isinstance(view_normalized, NormalizedView)
            or shape is None
            or len(shape) != 3
            or dim is None
            or index is None
            or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        ):
            return None
        selected_dim = dim.value % len(shape)
        structural_values = [dim, index]
        if selected_dim == len(shape) - 1:
            layout = local_reduce.grouped_tensors.get(view)
            if layout is None or layout.axis != 1:
                return None
            group, chunked = layout.group, False
        elif selected_dim == 1:
            structural_group = FlexGemmStructuralInt.from_value(shape[1])
            source_shape = tensor_meta_shape(view_normalized.source)
            if (
                structural_group is None
                or structural_group.value <= 1
                or source_shape is None
                or not statically_known_shape_equal(
                    (shape[0], structural_group.value * shape[2]), source_shape
                )
            ):
                return None
            group = structural_group.value
            structural_values.append(structural_group)
            chunked = True
        else:
            return None
        return OutputContractionUse(
            view_normalized.source,
            group,
            chunked,
            index.value,
            view,
            tuple(structural_values),
        )

    if not isinstance(normalized, NormalizedGetItem):
        return None
    split = normalized.source
    split_normalized = local_reduce.graph.normalized_nodes.get(split)
    if not isinstance(split_normalized, NormalizedSplit):
        return None
    index = FlexGemmStructuralInt.from_value(normalized.index)
    source = split_normalized.source
    split_size = FlexGemmStructuralInt.from_value(split_normalized.split_size)
    shape = tensor_meta_shape(source)
    if (
        index is None
        or shape is None
        or len(shape) != 2
        or not isinstance(shape[-1], int)
        or split_size is None
        or split_size.value <= 0
        or shape[-1] % split_size.value
        or split_normalized.dim not in (-1, 1)
        or not local_reduce.graph.depends_on(source, gemm)
    ):
        return None
    group = shape[-1] // split_size.value
    if group <= 1:
        return None
    return OutputContractionUse(
        source,
        group,
        True,
        index.value,
        split,
        (index, split_size),
    )


def build_output_contraction_plan(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionPlan | None:
    """Recognize a complete adjacent-N grouped main-output expression."""
    lanes: list[tuple[torch.fx.Node, OutputContractionUse]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    pending: list[Any] = [output]
    while pending:
        node = pending.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = match_output_contraction_use(node, gemm, local_reduce)
        if match is not None:
            lanes.append((node, match))
            continue
        if node is gemm or (
            node in local_reduce.grouped_tensors
            and local_reduce.graph.depends_on(node, gemm)
        ):
            return None
        pending.extend(reversed(tuple(iter_fx_node_inputs((node.args, node.kwargs)))))
    if not lanes:
        return None

    first = lanes[0][1]
    canonical_source = canonical_output_contraction_source(
        first.source, gemm, local_reduce
    )
    indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    layouts: dict[torch.fx.Node, GemmReductionGeometry] = {}
    structural_values: list[FlexGemmStructuralInt] = []
    for node, match in lanes:
        if (
            canonical_output_contraction_source(match.source, gemm, local_reduce)
            is not canonical_source
            or match.group != first.group
            or match.chunked != first.chunked
            or not -first.group <= match.index < first.group
        ):
            return None
        index = match.index % first.group
        indices.add(index)
        select_indices[node] = index
        layouts[match.layout_node] = GemmReductionGeometry(first.group, 1)
        structural_values.extend(match.structural_values)
    if indices != OrderedSet(range(first.group)):
        return None

    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if not isinstance(gemm_meta, torch.Tensor) or not isinstance(
        output_meta, torch.Tensor
    ):
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // first.group)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
    return OutputContractionPlan(
        FlexGemmOutputContraction(first.group, first.chunked),
        select_indices,
        layouts,
        tuple(structural_values),
    )
