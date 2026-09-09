# mypy: allow-untyped-defs
"""Captured aux tensor vectorization analysis for flex attention."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from enum import auto, Enum

import sympy

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing

from ...codegen.cutedsl.lane_analysis import classify_lane_expr
from ...ir import TensorBox
from ...virtualized import V


# Match one vectorized mask_mod call to one 32-bit R2P keep mask; on Blackwell,
# vec32 aux loads are close to vec16.
DEFAULT_MASK_MOD_VEC_SIZE = 32


class LoadKind(Enum):
    """How a captured aux tensor load behaves across vectorized KV lanes.

    GATHER: Cannot use a direct vector load.
    LANE_UNIFORM: The same value is used for every KV lane.
    CONTIGUOUS: Consecutive KV lanes map to contiguous storage.
    """

    GATHER = auto()
    LANE_UNIFORM = auto()
    CONTIGUOUS = auto()


@dataclasses.dataclass(frozen=True)
class AuxLoadVecInfo:
    """Vectorization classification for one captured aux tensor load.

    kind: Load classification.
    vec_size: Direct vector width for contiguous loads; None otherwise.
    """

    kind: LoadKind
    vec_size: int | None = None

    def __post_init__(self) -> None:
        """Validate that only contiguous loads carry a vector width."""
        if self.kind is LoadKind.CONTIGUOUS:
            if self.vec_size is None:
                raise AssertionError("CONTIGUOUS load requires a vec_size")
        else:
            if self.vec_size is not None:
                raise AssertionError(
                    f"non-CONTIGUOUS load must not carry vec_size, got {self.vec_size}"
                )

    @classmethod
    def gather(cls) -> "AuxLoadVecInfo":
        """Create a gather-load classification."""
        return cls(LoadKind.GATHER)

    @classmethod
    def lane_uniform(cls) -> "AuxLoadVecInfo":
        """Create a lane-uniform-load classification."""
        return cls(LoadKind.LANE_UNIFORM)

    @classmethod
    def contiguous(cls, vec_size: int) -> "AuxLoadVecInfo":
        """Create a contiguous direct-vector-load classification."""
        return cls(LoadKind.CONTIGUOUS, vec_size)


@dataclasses.dataclass(frozen=True)
class AuxVecPolicy:
    """ABI-specific rules for selecting an aux tensor vector width.

    q_idx_placeholder: Placeholder index for the query lane.
    kv_idx_placeholder: Placeholder index for the vectorized KV lane.
    max_vec_size: Largest direct captured-tensor load width to consider.
    min_index_rank_for_contiguous_load: Minimum indexed rank for direct vector loads.
        mask_mod uses rank 2 so 1-D KV masks remain scalar per-lane gathers;
        otherwise a short 1-D mask can shrink the packed-mask vector width.
    non_lane_placeholder_start: First non-lane placeholder participating in indices.
    """

    q_idx_placeholder: int
    kv_idx_placeholder: int
    max_vec_size: int
    min_index_rank_for_contiguous_load: int = 1
    non_lane_placeholder_start: int = 0


MASK_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=2,
    kv_idx_placeholder=3,
    max_vec_size=32,
    min_index_rank_for_contiguous_load=2,
)
SCORE_MOD_AUX_VEC_POLICY = AuxVecPolicy(
    q_idx_placeholder=3,
    kv_idx_placeholder=4,
    max_vec_size=8,
    non_lane_placeholder_start=1,
)


@dataclasses.dataclass(frozen=True)
class AuxIndexedTensor:
    """Captured tensor plus any safe partial indices already accumulated."""

    buffer: TensorBox
    indices: tuple[object, ...]


def make_fx_index_symbols(
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node] = (),
    *,
    kv_expr: sympy.Expr | None = None,
) -> tuple[sympy.Symbol, sympy.Symbol, dict[torch.fx.Node, sympy.Expr]]:
    """Build symbolic replacements for lane and non-lane FX index nodes."""
    q_idx = sympy.Symbol("q_idx", integer=True, nonnegative=True)
    kv_idx = sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    index_symbols = {
        node: sympy.Symbol(node.name, integer=True, nonnegative=True)
        for node in non_lane_index_nodes
    }
    index_symbols[q_idx_node] = q_idx
    index_symbols[kv_idx_node] = kv_idx if kv_expr is None else kv_expr
    return q_idx, kv_idx, index_symbols


def select_mask_mod_vec_size(
    *,
    has_mask_mod: bool,
    has_mask_aux_tensors: bool,
    supports_mask_mod_vec: bool,
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
) -> int | None:
    """Use mask_mod's (b, h, q, kv) ABI; vec32 matches the packed R2P mask width."""
    if not has_mask_mod or not supports_mask_mod_vec:
        return None
    if not has_mask_aux_tensors:
        return DEFAULT_MASK_MOD_VEC_SIZE

    vec_size = select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        MASK_MOD_AUX_VEC_POLICY,
    )
    return vec_size if vec_size > 1 else None


def select_score_mod_vec_size(
    *,
    has_score_mod: bool,
    has_aux_tensors: bool,
    is_sm100_or_later: bool,
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
) -> int | None:
    """Select score_mod vector width for captured aux loads.

    A return value of None means there are no captured tensor loads constraining
    score_mod.__vec_size__. For captured tensors, direct vector loads are capped
    to the SM100 CuTe score_mod aux-load path's supported width of 8.
    """
    if not has_score_mod or not has_aux_tensors:
        return None
    if not is_sm100_or_later:
        return 1
    return select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        SCORE_MOD_AUX_VEC_POLICY,
    )


def select_aux_mod_vec_size(
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
    policy: AuxVecPolicy,
) -> int:
    """Choose a safe vector width for captured tensor loads.

    The analysis follows aten.index chains from captured tensor placeholders,
    accumulates partial indices that do not depend on the vectorized KV lane,
    classifies complete loads, and picks the narrowest required contiguous
    width. Gather-like loads stay scalar per lane and can coexist with any
    vectorizable load that selects a non-scalar width.
    """
    if graph_module is None:
        return 1

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    num_fixed_placeholders = policy.kv_idx_placeholder + 1
    if len(placeholders) < num_fixed_placeholders:
        return 1

    aux_indexed_tensors = {
        placeholder: AuxIndexedTensor(buffer, ())
        for placeholder, buffer in zip(
            placeholders[num_fixed_placeholders:], other_buffers
        )
    }
    non_lane_index_nodes = placeholders[
        policy.non_lane_placeholder_start : policy.q_idx_placeholder
    ]
    selected_vec_size = policy.max_vec_size
    found_vectorizable_load = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.index.Tensor:
            continue
        buffer_node, indices = node.args
        if buffer_node not in aux_indexed_tensors:
            continue
        indexed_tensor = aux_indexed_tensors[buffer_node]
        if not isinstance(indices, (list, tuple)) or not indices:
            continue
        full_indices = indexed_tensor.indices + tuple(indices)
        rank = len(indexed_tensor.buffer.get_size())
        if len(full_indices) < rank:
            # Chained indexing is safe to track only until the KV lane is used.
            if is_safe_partial_aux_index(
                full_indices,
                placeholders[policy.q_idx_placeholder],
                placeholders[policy.kv_idx_placeholder],
                non_lane_index_nodes,
            ):
                aux_indexed_tensors[node] = AuxIndexedTensor(
                    indexed_tensor.buffer, full_indices
                )
            continue
        if len(full_indices) > rank:
            continue
        aux_load_vec_info = direct_aux_load_vec_size_and_kind(
            full_indices,
            indexed_tensor.buffer,
            placeholders[policy.q_idx_placeholder],
            placeholders[policy.kv_idx_placeholder],
            non_lane_index_nodes=non_lane_index_nodes,
            max_vec_size=policy.max_vec_size,
            min_index_rank_for_contiguous_load=policy.min_index_rank_for_contiguous_load,
        )
        match aux_load_vec_info.kind:
            case LoadKind.LANE_UNIFORM:
                found_vectorizable_load = True
            case LoadKind.GATHER:
                pass
            case LoadKind.CONTIGUOUS:
                contiguous_vec_size = aux_load_vec_info.vec_size
                if contiguous_vec_size is None:
                    raise AssertionError("CONTIGUOUS load must have a vec_size")
                selected_vec_size = min(selected_vec_size, contiguous_vec_size)
                found_vectorizable_load = True

    return selected_vec_size if found_vectorizable_load else 1


def is_safe_partial_aux_index(
    indices: tuple[object, ...],
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node],
) -> bool:
    """Return whether partial indices can be chained before the KV-lane index."""
    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    for index in indices:
        expr = fx_aux_index_to_sympy(index, index_symbols)
        if expr is None or kv_idx in expr.free_symbols:
            return False
    return True


def direct_aux_load_vec_size_and_kind(
    indices: object,
    buffer: TensorBox,
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node] = (),
    max_vec_size: int = 8,
    min_index_rank_for_contiguous_load: int = 1,
) -> AuxLoadVecInfo:
    """Return vector-load information for a captured aux load.

    The load is lane-uniform if none of its indices depend on KV. Otherwise,
    only the final indexed dimension may depend on KV, the storage for that
    dimension must be contiguous and vector-aligned, and the lane expression
    must enumerate consecutive values for the chosen vector width.
    """
    if not isinstance(indices, (list, tuple)) or not indices:
        return AuxLoadVecInfo.gather()
    if not (max_vec_size >= 2 and max_vec_size.bit_count() == 1):
        raise AssertionError(
            f"max_vec_size must be a power of two >= 2, got {max_vec_size}"
        )

    _, kv_idx, index_symbols = make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    index_exprs = [fx_aux_index_to_sympy(index, index_symbols) for index in indices]
    if any(expr is None for expr in index_exprs):
        return AuxLoadVecInfo.gather()
    if all(kv_idx not in expr.free_symbols for expr in index_exprs):
        return AuxLoadVecInfo.lane_uniform()

    last_expr = index_exprs[-1]
    if kv_idx not in last_expr.free_symbols:
        return AuxLoadVecInfo.gather()
    if len(indices) < min_index_rank_for_contiguous_load:
        return AuxLoadVecInfo.gather()

    prefix_exprs = index_exprs[:-1]
    if any(kv_idx in expr.free_symbols for expr in prefix_exprs):
        return AuxLoadVecInfo.gather()

    sizes = buffer.get_size()
    strides = buffer.get_stride()
    if not V.graph.sizevars.statically_known_equals(strides[-1], 1):
        return AuxLoadVecInfo.gather()

    # FlashAttention groups vectorized mod calls across consecutive KV lanes.
    offset = buffer.get_layout().offset
    vec_size = max_vec_size
    while vec_size >= 2:
        lane_info = classify_lane_expr(last_expr, kv_idx, max_width=vec_size)
        if (
            V.graph.sizevars.statically_known_multiple_of(sizes[-1], vec_size)
            and V.graph.sizevars.statically_known_multiple_of(offset, vec_size)
            and all(
                V.graph.sizevars.statically_known_multiple_of(stride, vec_size)
                for stride in strides[:-1]
            )
            and lane_info.is_contiguous
        ):
            return AuxLoadVecInfo.contiguous(vec_size)
        vec_size //= 2
    return AuxLoadVecInfo.gather()


def fx_aux_index_to_sympy(
    index: object,
    index_symbols: Mapping[torch.fx.Node, sympy.Expr],
    node_to_sympy: Callable[[torch.fx.Node], sympy.Expr | None] | None = None,
) -> sympy.Expr | None:
    """Translate the supported FX integer index operations to sympy."""
    if isinstance(index, int | sympy.Integer):
        return sympy.Integer(index)
    if not isinstance(index, torch.fx.Node):
        return None
    if index in index_symbols:
        return index_symbols[index]
    if node_to_sympy is not None:
        expr = node_to_sympy(index)
        if expr is not None:
            return expr
    if index.op != "call_function":
        return None

    args = index.args
    target = index.target
    match target:
        case torch.ops.aten.abs.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            return None if operand is None else sympy.Abs(operand)
        case torch.ops.aten.neg.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            return None if operand is None else -operand
        case torch.ops.aten.clamp.default:
            operand = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
            if operand is None:
                return None
            lo = args[1] if len(args) > 1 else index.kwargs.get("min")
            hi = args[2] if len(args) > 2 else index.kwargs.get("max")
            for bound, combine in ((lo, Max), (hi, Min)):
                if bound is not None:
                    bound_expr = fx_aux_index_to_sympy(
                        bound, index_symbols, node_to_sympy
                    )
                    if bound_expr is None:
                        return None
                    operand = combine(operand, bound_expr)
            return operand

    if len(args) < 2:
        return None
    lhs = fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
    rhs = fx_aux_index_to_sympy(args[1], index_symbols, node_to_sympy)
    if lhs is None or rhs is None:
        return None
    match target:
        case torch.ops.aten.add.Tensor | torch.ops.aten.add.Scalar:
            return V.graph.sizevars.simplify(lhs + rhs)
        case torch.ops.aten.sub.Tensor | torch.ops.aten.sub.Scalar:
            return V.graph.sizevars.simplify(lhs - rhs)
        case torch.ops.aten.mul.Tensor | torch.ops.aten.mul.Scalar:
            return V.graph.sizevars.simplify(lhs * rhs)
        case torch.ops.aten.minimum.default:
            return Min(lhs, rhs)
        case torch.ops.aten.maximum.default:
            return Max(lhs, rhs)
        case torch.ops.aten.remainder.Tensor | torch.ops.aten.remainder.Scalar:
            return ModularIndexing(lhs, 1, rhs)
        case torch.ops.aten.div.Tensor_mode if (
            index.kwargs.get("rounding_mode") == "floor"
        ):
            return FloorDiv(lhs, rhs)
        case _:
            return None
