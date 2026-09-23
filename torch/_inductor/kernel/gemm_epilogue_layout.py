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
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
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


if TYPE_CHECKING:
    from torch._inductor.kernel.gemm_epilogue_analysis import (
        GemmLocalReduceAnalysis,
        GemmOutputPlan,
    )


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
class FlexGemmLaneDomain:
    """Physical GEMM columns backing each column of a lane-derived value.

    Selecting one lane of an adjacent-N grouping multiplies ``physical_span``;
    ``chunked`` marks a first-level contiguous split, which QuACK realizes by
    interleaving B so the chunks become adjacent accumulator lanes. Values
    combine pointwise only within one domain, so each column keeps a single
    physical coordinate map no matter which lanes the expression reads.
    """

    physical_span: int
    chunked: bool


@dataclasses.dataclass(frozen=True)
class OutputContractionUse:
    """Describe one selected lane and the metadata the emitter lowers it with."""

    domain: FlexGemmLaneDomain
    group: int
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


def match_output_contraction_use(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionUse | None:
    """Match one interleaved select or contiguous split lane.

    An innermost select on a row-local grouping of the value's logical columns
    (``(M, N // span)``) nests inside an existing lane domain; contiguous chunks
    are recognized on physical GEMM columns only, since QuACK interleaves B.
    """
    gemm_shape = local_reduce.gemm_shape
    if gemm_shape is None or len(gemm_shape) != 2:
        return None
    normalized = local_reduce.graph.normalized_nodes.get(node)
    if isinstance(normalized, NormalizedSelect):
        view = normalized.source
        view_normalized = local_reduce.graph.normalized_nodes.get(view)
        dim = FlexGemmStructuralInt.from_value(normalized.dim)
        index = FlexGemmStructuralInt.from_value(normalized.index)
        if (
            not isinstance(view_normalized, NormalizedView)
            or len(view_normalized.shape) != 3
            or dim is None
            or index is None
            or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        ):
            return None
        source_domain = local_reduce.lane_domains.get(view_normalized.source)
        selected_dim = dim.value % 3
        if selected_dim == 2:
            group = FlexGemmStructuralInt.from_value(view_normalized.shape[-1])
            span = 1 if source_domain is None else source_domain.physical_span
            if (
                group is None
                or group.value <= 1
                or not _grouped_layout_matches_source_shape(
                    view_normalized.shape,
                    (gemm_shape[0], gemm_shape[1] // span),
                    GemmReductionGeometry(group.value, 1),
                )
            ):
                return None
            chunked = source_domain is not None and source_domain.chunked
        elif selected_dim == 1 and source_domain is None:
            shape = tensor_meta_shape(view)
            source_shape = tensor_meta_shape(view_normalized.source)
            if shape is None or source_shape is None:
                return None
            group = FlexGemmStructuralInt.from_value(shape[1])
            if (
                group is None
                or group.value <= 1
                or not statically_known_shape_equal(
                    (shape[0], group.value * shape[2]), source_shape
                )
            ):
                return None
            span, chunked = 1, True
        else:
            return None
        return OutputContractionUse(
            FlexGemmLaneDomain(span * group.value, chunked),
            group.value,
            index.value,
            view,
            (dim, index, group),
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
        or source in local_reduce.lane_domains
        or not local_reduce.graph.depends_on(source, gemm)
    ):
        return None
    group = shape[-1] // split_size.value
    if group <= 1:
        return None
    return OutputContractionUse(
        FlexGemmLaneDomain(group, True), group, index.value, split, (index, split_size)
    )


def build_output_contraction_plan(
    outputs: GemmOutputPlan,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionPlan | None:
    """Plan a main output computed in one lane domain of the GEMM columns."""
    output = outputs.output_storage or outputs.output
    domain = local_reduce.lane_domains.get(output)
    if domain is None:
        return None
    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if not isinstance(gemm_meta, torch.Tensor) or not isinstance(
        output_meta, torch.Tensor
    ):
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // domain.physical_span)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
    lanes = [
        (node, use)
        for node in (output, *local_reduce.graph.dependencies.get(output, ()))
        if (use := local_reduce.output_contraction_uses.get(node)) is not None
    ]
    return OutputContractionPlan(
        FlexGemmOutputContraction(domain.physical_span, domain.chunked),
        {node: use.index % use.group for node, use in lanes},
        {use.layout_node: GemmReductionGeometry(use.group, 1) for _, use in lanes},
        tuple(value for _, use in lanes for value in use.structural_values),
    )
