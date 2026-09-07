# mypy: allow-untyped-defs
"""Recognize grouped FlexGEMM layouts and reduction semantics.

PyTorch owns this FX-level analysis: grouped reshapes, pointwise dependency
tracking, reduction classification, and shape contracts. The EpiMod emitter
consumes these records while QuACK owns the physical grouped-reduction EpiOps.
"""

import dataclasses
from typing import Any

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    local_reduce_needs_physical_combine,
)
from torch._inductor.kernel.gemm_epilogue import iter_fx_node_inputs
from torch._inductor.kernel.gemm_epilogue_codegen import gemm_epilogue_cutedsl_op_name
from torch._inductor.kernel.gemm_epilogue_utils import (
    normalize_shape,
    statically_known_equal,
)
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch.fx.experimental.symbolic_shapes import (
    guard_int,
    has_guarding_hint,
    optimization_hint,
)


@dataclasses.dataclass(frozen=True)
class FlexGemmStructuralInt:
    """Hold a backed structural hint until an accepted match installs a guard."""

    value: int
    symbolic: torch.SymInt | None = None

    @classmethod
    def from_value(cls, value: Any) -> "FlexGemmStructuralInt | None":
        """Return a guard-free structural hint, rejecting unbacked values."""
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if isinstance(value, torch.SymInt):
            if not has_guarding_hint(value):
                return None
            return cls(optimization_hint(value), value)
        return cls(value) if isinstance(value, int) else None

    def guard(self) -> None:
        """Install the specialization guard after semantic validation succeeds."""
        if self.symbolic is not None and guard_int(self.symbolic) != self.value:
            raise AssertionError("FlexGEMM structural hint changed before commit")


@dataclasses.dataclass(frozen=True)
class FlexGemmTensorSSAFact:
    """Track logical values derived from fixed physical accumulator lanes."""

    root: torch.fx.Node
    physical_span: int
    chunked: bool
    lane_offsets: frozenset[int]
    storage_span: int = 1
    storage_offsets: frozenset[int] = frozenset((0,))
    reduced: bool = False
    external_tensor_inputs: frozenset[torch.fx.Node] = frozenset()

    @property
    def complete(self) -> bool:
        """Whether all physical lanes and stored logical slots are represented."""
        return self.lane_offsets == frozenset(
            range(self.physical_span)
        ) and self.storage_offsets == frozenset(range(self.storage_span))

    @property
    def output_span(self) -> int:
        """Physical accumulator columns represented by one stored output element."""
        return self.physical_span * self.storage_span


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout:
    """Describe a grouped M/N TensorSSA view inside the generated epilogue.

    Attributes:
        axis: GEMM output dimension being grouped: 0 for M, 1 for N.
        group_size: Number of contiguous output elements reduced as one group.
    """

    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    def fragment_group_size_expr(self, source: Any) -> str:
        """Return the grouped extent available in one TensorSSA fragment."""
        return (
            f"cutlass.const_expr(min({self.group_size}, "
            f"cute.size({source}.shape, mode=[0])))"
        )

    def fragment_repeat_expr(self, source: Any) -> str:
        """Return the number of grouped runs in one TensorSSA fragment."""
        return (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({self.group_size}, cute.size({source}.shape, mode=[0])))"
        )

    def tensorssa_shape(self, source: Any) -> str:
        """Return the grouped TensorSSA view for this logical axis."""
        group = self.fragment_group_size_expr(source)
        repeats = self.fragment_repeat_expr(source)
        if self.axis == 1:
            return f"((1, {group}, {repeats}), 1, 1)"
        return f"(({group}, 1, {repeats}), 1, 1)"

    def keepdim_shape(self, source: Any) -> str:
        """Return the reduced TensorSSA shape before fragment broadcast."""
        return f"((1, 1, {self.fragment_repeat_expr(source)}), 1, 1)"

    @property
    def reduction_profile(self) -> str:
        """Return the CuTe reduction profile for the grouped dimension."""
        return (
            "((None, 1, None), 1, 1)" if self.axis == 1 else "((1, None, None), 1, 1)"
        )

    def matches_reduction_dim(self, dim: Any) -> bool:
        """Return whether an FX reduction selects this layout's grouped dimension."""
        dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        return len(dims) == 1 and dims[0] in self.reduce_dims

    @property
    def needs_physical_combine(self) -> bool:
        return local_reduce_needs_physical_combine(self.axis, self.group_size)


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GroupedTensorSSALayout | None:
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
        return GroupedTensorSSALayout(axis=1, group_size=last.value)
    if (
        type(shape[-3]) is int
        and shape[-3] == -1
        and penultimate is not None
        and penultimate.value > 0
    ):
        return GroupedTensorSSALayout(axis=0, group_size=penultimate.value)
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
    layout: GroupedTensorSSALayout,
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
                and structural_group.value == layout.group_size
                and statically_known_equal(kept_m, m)
                and _group_count_matches_selected_dim(group_count, n, layout.group_size)
            )
        case 0, (group_count, group, kept_n):
            structural_group = FlexGemmStructuralInt.from_value(group)
            return (
                structural_group is not None
                and structural_group.value == layout.group_size
                and statically_known_equal(kept_n, n)
                and _group_count_matches_selected_dim(group_count, m, layout.group_size)
            )
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GroupedTensorSSALayout | None:
    """Recognize exact grouped M/N reshapes for the local-reduction contract."""
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
                    candidates.append(
                        GroupedTensorSSALayout(axis=1, group_size=group.value)
                    )
            if len(shape) >= 2:
                group = FlexGemmStructuralInt.from_value(shape[-2])
                if group is not None and group.value > 0:
                    candidates.append(
                        GroupedTensorSSALayout(axis=0, group_size=group.value)
                    )
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    (
        "_to_copy",
        "clamp",
        "clamp_max",
        "clamp_min",
        "convert_element_type",
        "inline_asm_elementwise",
    )
)


def tensor_meta_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return fake-tensor shape metadata when the FX value is tensor-like."""
    meta = node.meta.get("val")
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    return None


def node_preserves_tensor_shapes(node: torch.fx.Node) -> bool:
    """Reject pointwise broadcasts that cannot preserve a grouped TensorSSA input."""
    output_shape = tensor_meta_shape(node)
    if output_shape is None:
        return False
    output_shape_key = tuple(str(dim) for dim in output_shape)
    has_same_shape_input = False
    for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
        input_shape = tensor_meta_shape(input_node)
        if input_shape is None:
            continue
        input_shape_key = tuple(str(dim) for dim in input_shape)
        if input_shape_key == output_shape_key:
            has_same_shape_input = True
            continue
        try:
            broadcast_shape = get_broadcasted_shape(input_shape_key, output_shape_key)
        except AssertionError:
            return False
        if broadcast_shape is None or tuple(broadcast_shape) != output_shape_key:
            return False
    return has_same_shape_input


def is_pointwise_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    ) or gemm_epilogue_cutedsl_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    return is_pointwise_node(node) and node_preserves_tensor_shapes(node)


def view_or_reshape_args(node: torch.fx.Node) -> tuple[Any, tuple[Any, ...]] | None:
    if node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        shape = node.args[1]
        if isinstance(shape, (tuple, list, torch.Size)):
            return node.args[0], tuple(
                arg.meta.get("val", arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in shape
            )
    return None


def squeeze_source_node(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        return None
    source_node = node.args[0]
    return source_node if isinstance(source_node, torch.fx.Node) else None
