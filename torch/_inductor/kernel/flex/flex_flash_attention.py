# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import dataclasses
import functools
import importlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast, Literal, NamedTuple

import sympy
from sympy import Expr, Integer

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import V
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    SubgraphResults,
)


# Match one vectorized mask_mod call to one 32-bit R2P keep mask; on Blackwell,
# vec32 aux loads are close to vec16 and keep the packed-mask ABI simple.
DEFAULT_MASK_MOD_VEC_SIZE = 32
MAX_PACKED_MASK_INTERVALS = 8


@dataclasses.dataclass
class FlexFlashConfig:
    """Autotuning configuration for CuteDSL flex flash attention kernels.

    score_mod_vec_size: Number of elements processed per thread in the score_mod
        application loop. Maps to score_mod.__vec_size__ in CuTe flash attention.
        None uses the kernel default. Only effective for forward; backward does
        not currently support vectorized score_mod.
    mask_mod_vec_size: Number of consecutive KV lanes evaluated per mask_mod
        call. Maps to mask_mod.__mask_vec_size__ in CuTe flash attention and to
        the direct captured-tensor vector-load width for mask_mod.
    """

    score_mod_vec_size: int | None = None
    mask_mod_vec_size: int | None = None


class AuxLoadVecInfo(NamedTuple):
    vec_size: int | None
    is_direct_contiguous: bool


@dataclasses.dataclass(frozen=True)
class PackedMaskInterval:
    lower: str
    upper: str


@dataclasses.dataclass(frozen=True)
class LaneIndexAnalysis:
    q_idx: sympy.Symbol
    kv_idx: sympy.Symbol
    lane: sympy.Symbol

    @property
    def kv_lane(self) -> sympy.Expr:
        return self.kv_idx + self.lane


@dataclasses.dataclass(frozen=True)
class AuxIndexedTensor:
    buffer: TensorBox
    indices: tuple[object, ...]


def _make_fx_index_symbols(
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node] = (),
    *,
    kv_expr: sympy.Expr | None = None,
) -> tuple[sympy.Symbol, sympy.Symbol, dict[torch.fx.Node, sympy.Expr]]:
    q_idx = sympy.Symbol("q_idx", integer=True, nonnegative=True)
    kv_idx = sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    index_symbols = {
        node: sympy.Symbol(node.name, integer=True, nonnegative=True)
        for node in non_lane_index_nodes
    }
    index_symbols[q_idx_node] = q_idx
    index_symbols[kv_idx_node] = kv_idx if kv_expr is None else kv_expr
    return q_idx, kv_idx, index_symbols


def get_flex_flash_fwd_configs(
    has_score_mod: bool,
    has_aux_tensors: bool,
    device: torch.device | None = None,
    score_mod_graph_module: GraphModule | None = None,
    score_mod_other_buffers: Sequence[TensorBox] = (),
    has_mask_mod: bool = False,
    has_mask_aux_tensors: bool = False,
    mask_mod_graph_module: GraphModule | None = None,
    mask_mod_other_buffers: Sequence[TensorBox] = (),
) -> list[FlexFlashConfig]:
    cuda_major = None
    if torch.cuda.is_available() and (
        has_mask_mod or (has_score_mod and has_aux_tensors)
    ):
        device_index = None if device is None else device.index
        cuda_major = torch.cuda.get_device_capability(device_index)[0]
    mask_mod_vec_size = select_mask_mod_vec_size(
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        supports_mask_mod_vec=cuda_major in (10, 11),
        graph_module=mask_mod_graph_module,
        other_buffers=mask_mod_other_buffers,
    )
    score_mod_vec_size = select_score_mod_vec_size(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_aux_tensors,
        is_sm100_or_later=cuda_major is not None and cuda_major >= 10,
        graph_module=score_mod_graph_module,
        other_buffers=score_mod_other_buffers,
    )

    if (
        has_score_mod
        and score_mod_vec_size is None
        and torch._inductor.config.max_autotune
    ):
        configs = [
            FlexFlashConfig(score_mod_vec_size=v, mask_mod_vec_size=mask_mod_vec_size)
            for v in (1, 2, 4, 8, 16, 32, 64, 128)
        ]
    else:
        configs = [
            FlexFlashConfig(
                score_mod_vec_size=score_mod_vec_size,
                mask_mod_vec_size=mask_mod_vec_size,
            )
        ]
    max_configs = torch._inductor.config.test_configs.max_flex_configs
    if max_configs is not None and len(configs) > max_configs:
        configs = configs[:max_configs]
    return configs


def select_mask_mod_vec_size(
    *,
    has_mask_mod: bool,
    has_mask_aux_tensors: bool,
    supports_mask_mod_vec: bool,
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
) -> int | None:
    if not has_mask_mod or not supports_mask_mod_vec:
        return None
    if not has_mask_aux_tensors:
        return DEFAULT_MASK_MOD_VEC_SIZE

    vec_size = _select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        q_idx_placeholder=2,
        kv_idx_placeholder=3,
        max_vec_size=32,
        min_index_rank_for_contiguous_load=2,
        allow_gather_loads=True,
        require_contiguous_load=True,
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
    if not has_score_mod or not has_aux_tensors:
        return None
    if not is_sm100_or_later:
        return 1
    return _select_aux_mod_vec_size(
        graph_module,
        other_buffers,
        q_idx_placeholder=3,
        kv_idx_placeholder=4,
        max_vec_size=8,
        non_lane_placeholder_start=1,
    )


def _select_aux_mod_vec_size(
    graph_module: GraphModule | None,
    other_buffers: Sequence[TensorBox],
    *,
    q_idx_placeholder: int,
    kv_idx_placeholder: int,
    max_vec_size: int,
    min_index_rank_for_contiguous_load: int = 1,
    allow_gather_loads: bool = False,
    require_contiguous_load: bool = False,
    non_lane_placeholder_start: int = 0,
) -> int:
    """Choose a safe vector width for captured tensor loads.

    Vectorization is enabled from captured tensor loads that can be emitted as
    direct contiguous vector loads or lane-uniform scalar loads. By default, any
    load requiring gather semantics disables vectorization. Mask mods can allow
    gather loads so one vectorizable aux load still enables packed/R2P mask
    application while unrelated aux loads stay on the per-lane gather path.
    """
    if graph_module is None:
        return 1

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    num_fixed_placeholders = kv_idx_placeholder + 1
    if len(placeholders) < num_fixed_placeholders:
        return 1

    aux_indexed_tensors = {
        placeholder: AuxIndexedTensor(buffer, ())
        for placeholder, buffer in zip(
            placeholders[num_fixed_placeholders:], other_buffers
        )
    }
    non_lane_index_nodes = placeholders[non_lane_placeholder_start:q_idx_placeholder]
    selected_vec_size = max_vec_size
    found_vectorizable_load = False
    found_contiguous_load = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.index.Tensor:
            continue
        buffer_node, indices = node.args
        if buffer_node not in aux_indexed_tensors:
            continue
        indexed_tensor = aux_indexed_tensors[buffer_node]
        if not isinstance(indices, (list, tuple)) or not indices:
            if allow_gather_loads:
                continue
            return 1
        full_indices = indexed_tensor.indices + tuple(indices)
        rank = len(indexed_tensor.buffer.get_size())
        if len(full_indices) < rank:
            if _is_safe_partial_aux_index(
                full_indices,
                placeholders[q_idx_placeholder],
                placeholders[kv_idx_placeholder],
                non_lane_index_nodes,
            ):
                aux_indexed_tensors[node] = AuxIndexedTensor(
                    indexed_tensor.buffer, full_indices
                )
            elif not allow_gather_loads:
                return 1
            continue
        if len(full_indices) > rank:
            if allow_gather_loads:
                continue
            return 1
        aux_load_vec_info = direct_aux_load_vec_size_and_kind(
            full_indices,
            indexed_tensor.buffer,
            placeholders[q_idx_placeholder],
            placeholders[kv_idx_placeholder],
            non_lane_index_nodes=non_lane_index_nodes,
            max_vec_size=max_vec_size,
            min_index_rank_for_contiguous_load=min_index_rank_for_contiguous_load,
        )
        if aux_load_vec_info.vec_size is None:
            if allow_gather_loads:
                continue
            return 1
        selected_vec_size = min(selected_vec_size, aux_load_vec_info.vec_size)
        found_vectorizable_load = True
        found_contiguous_load = (
            found_contiguous_load or aux_load_vec_info.is_direct_contiguous
        )

    if require_contiguous_load and not found_contiguous_load:
        return 1
    return selected_vec_size if found_vectorizable_load else 1


def _is_safe_partial_aux_index(
    indices: tuple[object, ...],
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    non_lane_index_nodes: Sequence[torch.fx.Node],
) -> bool:
    _, kv_idx, index_symbols = _make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    for index in indices:
        expr = _fx_aux_index_to_sympy(index, index_symbols)
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

    vec_size is the largest safe KV-lane vector width, or None if the load cannot
    use a direct vector/scalar load. is_direct_contiguous is True only when the
    vector lanes map to consecutive memory locations; lane-uniform scalar loads
    return False. non_lane_index_nodes are placeholders such as batch/head that
    may appear in non-KV prefix dimensions. min_index_rank_for_contiguous_load
    excludes lower-rank loads from contiguous-vector consideration while still
    allowing uniform loads.
    """
    if not isinstance(indices, (list, tuple)) or not indices:
        return AuxLoadVecInfo(None, False)
    assert max_vec_size >= 2 and max_vec_size.bit_count() == 1

    _, kv_idx, index_symbols = _make_fx_index_symbols(
        q_idx_node, kv_idx_node, non_lane_index_nodes
    )
    index_exprs = [_fx_aux_index_to_sympy(index, index_symbols) for index in indices]
    if any(expr is None for expr in index_exprs):
        return AuxLoadVecInfo(None, False)
    if all(kv_idx not in expr.free_symbols for expr in index_exprs):
        return AuxLoadVecInfo(max_vec_size, False)

    last_expr = index_exprs[-1]
    if kv_idx not in last_expr.free_symbols:
        return AuxLoadVecInfo(None, False)
    if len(indices) < min_index_rank_for_contiguous_load:
        return AuxLoadVecInfo(None, False)

    prefix_exprs = index_exprs[:-1]
    if any(kv_idx in expr.free_symbols for expr in prefix_exprs):
        return AuxLoadVecInfo(None, False)

    sizes = buffer.get_size()
    strides = buffer.get_stride()
    if not V.graph.sizevars.statically_known_equals(strides[-1], 1):
        return AuxLoadVecInfo(None, False)

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
            return AuxLoadVecInfo(vec_size, True)
        vec_size //= 2
    return AuxLoadVecInfo(None, False)


def _fx_aux_index_to_sympy(
    index: object,
    index_symbols: Mapping[torch.fx.Node, sympy.Expr],
    node_to_sympy: Callable[[torch.fx.Node], sympy.Expr | None] | None = None,
) -> sympy.Expr | None:
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
    if len(args) < 2:
        return None
    lhs = _fx_aux_index_to_sympy(args[0], index_symbols, node_to_sympy)
    rhs = _fx_aux_index_to_sympy(args[1], index_symbols, node_to_sympy)
    if lhs is None or rhs is None:
        return None
    match target:
        case torch.ops.aten.add.Tensor | torch.ops.aten.add.Scalar:
            return V.graph.sizevars.simplify(lhs + rhs)
        case torch.ops.aten.sub.Tensor | torch.ops.aten.sub.Scalar:
            return V.graph.sizevars.simplify(lhs - rhs)
        case torch.ops.aten.mul.Tensor | torch.ops.aten.mul.Scalar:
            return V.graph.sizevars.simplify(lhs * rhs)
        case torch.ops.aten.remainder.Tensor | torch.ops.aten.remainder.Scalar:
            return ModularIndexing(lhs, 1, rhs)
        case torch.ops.aten.div.Tensor_mode if (
            index.kwargs.get("rounding_mode") == "floor"
        ):
            return FloorDiv(lhs, rhs)
        case _:
            return None


def _get_flex_flash_bwd_configs() -> list[FlexFlashConfig]:
    return [FlexFlashConfig()]


aten = torch.ops.aten
prims = torch.ops.prims


@functools.lru_cache(maxsize=1)
def ensure_flash_available() -> bool:
    """Check if flash-attn is importable; cache the result for reuse.

    Call ensure_flash_available.cache_clear() after installing flash-attn
    in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None  # type: ignore[attr-defined]
    except ImportError:
        return False


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from ...codegen.cutedsl.lane_analysis import (
    classify_lane_expr,
    decompose_affine_lane_expr,
)


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_flex_template("flash_attention")
)
flash_attention_backward_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_backward_cutedsl",
    source=load_flex_template("flash_attention_backward"),
)


class HierarchicalIndex(sympy.Function):
    """
    Inert wrapper to carry an N-D index tuple through Inductor's SymPy-based IR.

    Inductor generally represents a tensor index as a single `sympy.Expr` (often a
    flattened linear offset in memory). CuteDSL, however, wants structured coordinates so it
    can emit `tensor[i, j, ...]` and handle strides internally. We therefore wrap
    the per-dimension indices in a `sympy.Function` node: this keeps the value a
    `sympy.Expr` for existing substitution/CSE machinery, while letting CuteDSL
    codegen pattern-match and unpack the coordinates via `index.args`.

    `eval()` returns None to keep the node inert (no simplification/flattening).

    These nodes are intended to be short-lived wrappers and are only interpreted by
    CuteDSL codegen (see `ModificationWrapperCuteDSL.load` in
    `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`).
    """

    @classmethod
    def eval(cls, *args):
        return None


def _hierarchical_indexer_cute(
    size: Sequence[int],
    stride: Sequence[int] | None = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[Expr]], Expr]:
    """Return an indexer that preserves multi-dimensional indices for CuteDSL."""

    def indexer(indices: Sequence[Expr]) -> Expr:
        assert offset == Integer(0), "Offset not supported for hierarchical indexing"
        assert len(indices) == len(size), (
            f"Rank mismatch: got {len(indices)} indices for tensor of rank {len(size)}"
        )
        if not indices:
            return Integer(0)
        if len(indices) == 1:
            return indices[0]
        return HierarchicalIndex(*indices)

    return indexer


@contextmanager
def patch_fixed_layout_indexer_for_cutedsl():
    """
    Temporarily swap FixedLayout.make_indexer so CuteDSL sees hierarchical indexing.

    Note [CuteDSL indexer patch]:
    Flex flash attention only supports a limited set of IR ops (pointwise, reads, no stores),
    so temporarily changing the indexing behavior is safe for the kernels we emit today.
    TODO(dynamic shapes): Reconfirm once flex flash attention supports dynamic shapes.
    """
    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self):
        return _hierarchical_indexer_cute(self.size, self.stride, self.offset)

    FixedLayout.make_indexer = cutedsl_make_indexer  # type: ignore[assignment]
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer  # type: ignore[assignment]


def wrap_choice_render_with_cutedsl_indexer(choice: Any) -> None:
    """
    Wrap a template choice's kernel render to apply CuteDSL indexer patching.

    See Note [CuteDSL indexer patch]:
    CuteDSL handles tensor strides internally, so template rendering must use
    hierarchical indexing.
    """
    original_make_kernel_render = choice.make_kernel_render

    def make_kernel_render_with_patch(*args, **kwargs):
        render_kernel, render = original_make_kernel_render(*args, **kwargs)

        def render_with_patch():
            with patch_fixed_layout_indexer_for_cutedsl():
                return render()

        return render_kernel, render_with_patch

    choice.make_kernel_render = make_kernel_render_with_patch


def input_buffers_require_grads(graph_module, num_score_mod_placeholders: int):
    """Check if any of the input buffers (beyond the score mod placeholders) require gradients."""
    inputs = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node)
    if len(inputs) <= num_score_mod_placeholders:
        return False

    def requires_grad(n):
        tensor_meta = n.meta.get("tensor_meta")
        return tensor_meta.requires_grad if tensor_meta is not None else False

    return any(requires_grad(n) for n in inputs[num_score_mod_placeholders:])


def is_trivial_score_graph(graph_module: GraphModule) -> bool:
    """Backwards currently doesn't support score_mods, match against identity"""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    assert len(output) == 1, "Got graph w/ multiple outputs"
    output_val = output[0].args[0]
    # The identity graph just sends the score straight through
    return output_val == placeholders[0]


def is_trivial_mask_graph(graph_module: GraphModule) -> bool:
    """Mask graph is trivial when it only gates via the default full op."""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    assert len(output) == 1, "Got graph w/ multiple outputs"
    output_val = output[0].args[0]

    # mask mod graph is empty if we have 4 inputs and full_default output
    return len(placeholders) == 4 and output_val.target is torch.ops.aten.full.default


def is_bool_full_node(node: torch.fx.Node, value: bool) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.full.default
        and len(node.args) >= 2
        and node.args[0] == []
        and node.args[1] is value
    )


def sympy_to_cute_index(
    expr: sympy.Expr, symbol_codes: Mapping[sympy.Symbol, str] | None = None
) -> str | None:
    expr = V.graph.sizevars.simplify(expr)
    if isinstance(expr, sympy.Integer):
        return f"cutlass.Int32({int(expr)})"
    if isinstance(expr, sympy.Symbol):
        if symbol_codes is not None and expr in symbol_codes:
            return symbol_codes[expr]
        if expr.name == "q_idx":
            return "q_idx[0]"
        if expr.name == "kv_idx":
            return "kv_idx[0]"
    if isinstance(expr, FloorDiv):
        lhs, rhs = (sympy_to_cute_index(arg, symbol_codes) for arg in expr.args)
        if lhs is not None and rhs is not None:
            return f"({lhs} // {rhs})"
        return None
    if isinstance(expr, sympy.Add):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        return "(" + " + ".join(args) + ")"
    if isinstance(expr, sympy.Mul):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        return "(" + " * ".join(args) + ")"
    return None


def is_aten_index_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.index.Tensor


def fx_node_dtype(expr: torch.fx.Node) -> torch.dtype | None:
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tensor_meta.dtype
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    if is_aten_index_node(expr):
        base = expr.args[0]
        if isinstance(base, torch.fx.Node):
            return fx_node_dtype(base)
    return None


def fx_node_shape(expr: torch.fx.Node) -> tuple[int, ...] | None:
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tuple(tensor_meta.shape)
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return None


@dataclasses.dataclass
class PackedMaskAnalyzer:
    """Lower one Flex mask_mod FX graph to packed 32-lane mask intervals.

    The analyzer owns the mask graph signature, symbolic q/kv/lane variables,
    and any rendered CuteDSL code for q-uniform aux tensor bounds. Call
    ``node_to_intervals`` on the mask graph output; unsupported expressions
    return ``None`` so the caller can use the generic mask lowering path.
    """

    placeholders: Sequence[torch.fx.Node]
    q_idx: torch.fx.Node
    kv_idx: torch.fx.Node
    lane_analysis: LaneIndexAnalysis
    symbol_codes: dict[sympy.Symbol, str] = dataclasses.field(default_factory=dict)
    aux_load_symbols: dict[torch.fx.Node, sympy.Symbol] = dataclasses.field(
        default_factory=dict
    )
    next_symbol_id: int = 0

    def q_uniform_cute_expr(
        self,
        expr: object,
        *,
        for_index: bool = False,
        index_dim_size: int | sympy.Expr | None = None,
    ) -> str | None:
        """Return CuteDSL code for a scalar expression that is uniform across q lanes."""
        if isinstance(expr, int | sympy.Integer):
            index = int(expr)
            if for_index and index < 0:
                if index_dim_size is None:
                    return None
                index = V.graph.sizevars.guard_int(index + index_dim_size)
            return f"cutlass.Int32({index})"
        if not isinstance(expr, torch.fx.Node):
            return None
        if expr is self.q_idx:
            return "q_idx[0]"
        if expr is self.kv_idx:
            return None
        if len(self.placeholders) >= 2 and expr is self.placeholders[0]:
            return "b_idx[0]"
        if len(self.placeholders) >= 2 and expr is self.placeholders[1]:
            return "h_idx[0]"
        if expr in self.placeholders[4:]:
            return f"aux_tensors[{self.placeholders[4:].index(expr)}]"
        if expr.op != "call_function":
            return None

        if is_aten_index_node(expr):
            return self._q_uniform_index_cute_expr(
                expr, for_index=for_index, index_dim_size=index_dim_size
            )

        args = expr.args
        if len(args) < 2:
            return None
        lhs = self.q_uniform_cute_expr(args[0])
        rhs = self.q_uniform_cute_expr(args[1])
        if lhs is None or rhs is None:
            return None
        match expr.target:
            case torch.ops.aten.add.Tensor | torch.ops.aten.add.Scalar:
                return f"({lhs} + {rhs})"
            case torch.ops.aten.sub.Tensor | torch.ops.aten.sub.Scalar:
                return f"({lhs} - {rhs})"
            case torch.ops.aten.mul.Tensor | torch.ops.aten.mul.Scalar:
                return f"({lhs} * {rhs})"
            case torch.ops.aten.remainder.Tensor | torch.ops.aten.remainder.Scalar:
                return f"({lhs} % {rhs})"
            case torch.ops.aten.div.Tensor_mode if (
                expr.kwargs.get("rounding_mode") == "floor"
            ):
                return f"({lhs} // {rhs})"
            case _:
                return None

    def _q_uniform_index_cute_expr(
        self,
        expr: torch.fx.Node,
        *,
        for_index: bool,
        index_dim_size: int | sympy.Expr | None,
    ) -> str | None:
        """Render a q-uniform aux tensor index expression.

        For example, ``offsets[doc_ids[b, q_idx]]`` becomes a nested CuteDSL
        load from ``aux_tensors``. Dynamic aux-derived indices are wrapped for
        negative values before indexing to preserve PyTorch index semantics.
        """
        if fx_node_dtype(expr) not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return None
        result_shape = fx_node_shape(expr)
        if result_shape is not None and len(result_shape) != 0:
            return None
        base, indices = expr.args
        base_code = self.q_uniform_cute_expr(base)
        base_shape = fx_node_shape(base) if isinstance(base, torch.fx.Node) else None
        if base_code is None or not isinstance(indices, (list, tuple)):
            return None
        if base_shape is None or len(indices) != len(base_shape):
            return None
        index_codes = []
        for dim, index in enumerate(indices):
            dim_size = base_shape[dim]
            index_code = self.q_uniform_cute_expr(
                index, for_index=True, index_dim_size=dim_size
            )
            if index_code is None:
                return None
            if (
                dim_size is not None
                and isinstance(index, torch.fx.Node)
                and index
                not in (self.q_idx, self.placeholders[0], self.placeholders[1])
            ):
                dtype = fx_node_dtype(index)
                integer_type = "Int64" if dtype == torch.int64 else "Int32"
                size_code = f"cutlass.{integer_type}({dim_size})"
                zero_code = f"cutlass.{integer_type}(0)"
                index_code = (
                    f"({index_code} + {size_code} "
                    f"if {index_code} < {zero_code} else {index_code})"
                )
            index_codes.append(index_code)
        load = f"{base_code}[{', '.join(index_codes)}]"
        if for_index and fx_node_dtype(expr) == torch.int64:
            return load
        return f"cutlass.Int32({load})"

    def fx_mask_expr_to_sympy(self, expr: object) -> sympy.Expr | None:
        """Convert mask arithmetic to SymPy, replacing aux loads with symbols."""
        _, _, index_symbols = _make_fx_index_symbols(
            self.q_idx, self.kv_idx, kv_expr=self.lane_analysis.kv_lane
        )
        return _fx_aux_index_to_sympy(expr, index_symbols, self.mask_aux_load_to_symbol)

    def mask_aux_load_to_symbol(self, node: torch.fx.Node) -> sympy.Expr | None:
        """Assign a stable symbolic bound name to a q-uniform aux load."""
        if not is_aten_index_node(node):
            return None
        if node in self.aux_load_symbols:
            return self.aux_load_symbols[node]
        q_uniform_code = self.q_uniform_cute_expr(node)
        if q_uniform_code is None:
            return None
        symbol = sympy.Symbol(f"mask_bound_{self.next_symbol_id}", integer=True)
        self.next_symbol_id += 1
        self.symbol_codes[symbol] = q_uniform_code
        self.aux_load_symbols[node] = symbol
        return symbol

    def merge_intersection(
        self,
        intervals: tuple[PackedMaskInterval, ...],
        new_intervals: tuple[PackedMaskInterval, ...],
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Intersect interval unions, falling back if the cross-product grows too large."""
        if len(intervals) * len(new_intervals) > MAX_PACKED_MASK_INTERVALS:
            return None
        merged = []
        for lhs in intervals:
            for rhs in new_intervals:
                merged.append(
                    PackedMaskInterval(
                        f"max({lhs.lower}, {rhs.lower})",
                        f"min({lhs.upper}, {rhs.upper})",
                    )
                )
        return tuple(merged)

    def lane_comparison_to_intervals(
        self,
        lhs_expr: sympy.Expr,
        rhs_expr: sympy.Expr,
        *,
        strict: bool,
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Lower an affine lane comparison into one packed mask interval."""
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_analysis.lane)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff == 1:
            upper = sympy_to_cute_index(
                -rest if strict else -rest + 1, self.symbol_codes
            )
            if upper is not None:
                return (PackedMaskInterval("cutlass.Int32(0)", upper),)
            return None
        if lane_coeff == -1:
            lower = sympy_to_cute_index(rest + 1 if strict else rest, self.symbol_codes)
            if lower is not None:
                return (PackedMaskInterval(lower, "cutlass.Int32(32)"),)
            return None
        return None

    def lane_equality_to_intervals(
        self, lhs_expr: sympy.Expr, rhs_expr: sympy.Expr
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Lower lane equality into a singleton or block-aligned interval."""
        if isinstance(lhs_expr, FloorDiv) and isinstance(rhs_expr, FloorDiv):
            intervals = self._floor_div_equality_to_intervals(lhs_expr, rhs_expr)
            if intervals is not None:
                return intervals
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_analysis.lane)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff in (1, -1):
            lane_value = -rest if lane_coeff == 1 else rest
            lower = sympy_to_cute_index(lane_value, self.symbol_codes)
            upper = sympy_to_cute_index(lane_value + 1, self.symbol_codes)
            if lower is not None and upper is not None:
                return (PackedMaskInterval(lower, upper),)
            return None
        return None

    def _floor_div_equality_to_intervals(
        self, lhs_expr: FloorDiv, rhs_expr: FloorDiv
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Lower block equality like ``q_idx // B == kv_idx // B`` to ``[block_start, block_start + B)``."""
        lhs_base, lhs_divisor = lhs_expr.args
        rhs_base, rhs_divisor = rhs_expr.args
        if lhs_divisor != rhs_divisor:
            return None
        block_start = None
        if (
            lhs_base == self.lane_analysis.q_idx
            and rhs_base == self.lane_analysis.kv_lane
        ):
            block_start = V.graph.sizevars.simplify(
                lhs_expr * lhs_divisor - self.lane_analysis.kv_idx
            )
        elif (
            rhs_base == self.lane_analysis.q_idx
            and lhs_base == self.lane_analysis.kv_lane
        ):
            block_start = V.graph.sizevars.simplify(
                rhs_expr * rhs_divisor - self.lane_analysis.kv_idx
            )
        if block_start is None:
            return None
        lower = sympy_to_cute_index(block_start, self.symbol_codes)
        upper = sympy_to_cute_index(block_start + lhs_divisor, self.symbol_codes)
        if lower is not None and upper is not None:
            return (PackedMaskInterval(lower, upper),)
        return None

    def comparison_to_intervals(
        self,
        lhs: object,
        rhs: object,
        *,
        strict: bool,
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Convert FX comparison operands to SymPy before interval lowering."""
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return self.lane_comparison_to_intervals(lhs_expr, rhs_expr, strict=strict)

    def equality_to_intervals(
        self, lhs: object, rhs: object
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Convert FX equality operands to SymPy before interval lowering."""
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return self.lane_equality_to_intervals(lhs_expr, rhs_expr)

    def node_to_intervals(
        self, node: torch.fx.Node
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Recursively lower a boolean mask node into a union of intervals."""
        if is_bool_full_node(node, True):
            return (PackedMaskInterval("cutlass.Int32(0)", "cutlass.Int32(32)"),)
        if is_bool_full_node(node, False):
            return ()
        if node.op != "call_function":
            return None
        match node.target:
            case torch.ops.aten.bitwise_and.Tensor | torch.ops.aten.logical_and.default:
                return self.combine_binary_intervals(node, intersect=True)
            case torch.ops.aten.bitwise_or.Tensor | torch.ops.aten.logical_or.default:
                return self.combine_binary_intervals(node, intersect=False)
            case torch.ops.aten.le.Tensor | torch.ops.aten.le.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=False
                )
            case torch.ops.aten.lt.Tensor | torch.ops.aten.lt.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=True
                )
            case torch.ops.aten.ge.Tensor | torch.ops.aten.ge.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=False
                )
            case torch.ops.aten.gt.Tensor | torch.ops.aten.gt.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=True
                )
            case torch.ops.aten.eq.Tensor | torch.ops.aten.eq.Scalar:
                return self.equality_to_intervals(node.args[0], node.args[1])
            case _:
                return None

    def combine_binary_intervals(
        self, node: torch.fx.Node, *, intersect: bool
    ) -> tuple[PackedMaskInterval, ...] | None:
        """Lower AND/OR by combining the recursively lowered child interval unions."""
        lhs, rhs = node.args
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            return None
        lhs_intervals = self.node_to_intervals(lhs)
        rhs_intervals = self.node_to_intervals(rhs)
        if lhs_intervals is None or rhs_intervals is None:
            return None
        if intersect:
            return self.merge_intersection(lhs_intervals, rhs_intervals)
        if len(lhs_intervals) + len(rhs_intervals) > MAX_PACKED_MASK_INTERVALS:
            return None
        return lhs_intervals + rhs_intervals


def select_packed_mask_intervals(
    graph_module: GraphModule,
) -> tuple[PackedMaskInterval, ...] | None:
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    output = [node for node in nodes if node.op == "output"]
    if len(placeholders) < 4 or len(output) != 1:
        return None

    q_idx, kv_idx = placeholders[2], placeholders[3]
    output_val = output[0].args[0]
    if not isinstance(output_val, torch.fx.Node):
        return None

    lane_analysis = LaneIndexAnalysis(
        q_idx=sympy.Symbol("q_idx", integer=True, nonnegative=True),
        kv_idx=sympy.Symbol("kv_idx", integer=True, nonnegative=True),
        lane=sympy.Symbol("mask_lane", integer=True, nonnegative=True),
    )
    analyzer = PackedMaskAnalyzer(
        placeholders=placeholders,
        q_idx=q_idx,
        kv_idx=kv_idx,
        lane_analysis=lane_analysis,
    )
    intervals = analyzer.node_to_intervals(output_val)
    if intervals is None:
        return None
    if intervals == (PackedMaskInterval("cutlass.Int32(0)", "cutlass.Int32(32)"),):
        return None
    return intervals


@functools.lru_cache(maxsize=1)
def _is_symbol_from_tensor_shape(symbol: sympy.Symbol, shape_env: Any) -> bool:
    from torch._dynamo.source import TensorPropertySource

    sources = shape_env.var_to_sources.get(symbol, [])
    return any(isinstance(s, TensorPropertySource) for s in sources)


def has_unsupported_captured_scalars(
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> bool:
    """Return True when FLASH captures dynamic scalars it cannot inline."""
    shape_env = V.graph.sizevars.shape_env

    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        if isinstance(buf, sympy.Expr):
            for symbol in buf.free_symbols:
                if not _is_symbol_from_tensor_shape(symbol, shape_env):
                    return True
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and device.type == "cpu" and len(size) == 0:
                return True
    return False


def _can_use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    num_score_mod_placeholders: int,
) -> tuple[bool, str]:
    """Check if flex flash attention can be used for the given inputs.

    Returns:
        tuple: (can_use, reason) where reason explains why it can't be used if can_use is False
    """
    if not ensure_flash_available():
        return False, "CUTE flash attention library is not available"

    if input_buffers_require_grads(subgraph.graph_module, num_score_mod_placeholders):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention)",
        )

    return True, ""


def _use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    kernel_options: dict[str, Any],
    num_score_mod_placeholders: int,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
) -> bool:
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        subgraph: The score modification subgraph
        mask_graph: The mask modification subgraph
        kernel_options: Kernel configuration options
        num_score_mod_placeholders: Number of placeholders in score_mod
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention(
        subgraph,
        mask_graph,
        num_score_mod_placeholders,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    block_mask: tuple[Any, ...],
    scale: float,
    kernel_options: dict[str, Any],
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    score_mod_other_buffers: list[TensorBox],
    mask_mod_other_buffers: list[TensorBox],
    kv_num_blocks: TensorBox | None,
    kv_indices: TensorBox | None,
    full_kv_num_blocks: TensorBox | None,
    full_kv_indices: TensorBox | None,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    mask_graph: Subgraph,
    subgraph: Subgraph | None = None,
) -> tuple[TensorBox, TensorBox]:
    """Create a flex flash attention kernel using CuteDSL template."""
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Mixed query, key, and value dtype is not supported on this platform, "
            f"got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype}."
        )
    if not ensure_flash_available():
        raise RuntimeError("CUTE flash attention not available")

    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()
    assert device is not None, "Device must be specified"

    # Match stride pattern from query tensor
    q_strides = query.get_stride()
    out_size = [batch_size, num_heads, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    output = empty_strided(
        size=out_size,
        stride=out_strides,
        dtype=dtype,
        device=device,
    )

    lse = empty_strided(
        size=[batch_size, num_heads, seq_len_q],
        stride=None,  # LSE can be contiguous
        dtype=torch.float32,  # LSE is always fp32
        device=device,
    )

    # Create layout for primary output
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in output.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    mask_graph_is_trivial = is_trivial_mask_graph(mask_graph.graph_module)
    score_graph_is_trivial = subgraph is None or is_trivial_score_graph(
        subgraph.graph_module
    )

    needs_block_mask = not mask_graph_is_trivial
    has_score_mod = not score_graph_is_trivial
    has_full_blocks = full_kv_num_blocks is not None

    choices: list[Any] = []
    assert flash_attention_cutedsl_template is not None

    input_nodes = [query, key, value, lse]
    if has_full_blocks:
        input_nodes.extend(
            [kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices]
        )

    if needs_block_mask and not has_full_blocks:
        raise NotImplementedError(
            "Flash attention with block mask but without full blocks is not supported yet"
        )

    subgraphs = []
    if has_score_mod:
        subgraphs.append(subgraph_buffer)
    subgraphs.append(mask_graph_buffer)

    configs = get_flex_flash_fwd_configs(
        has_score_mod=has_score_mod,
        has_aux_tensors=len(score_mod_other_buffers) > 0,
        device=device,
        score_mod_graph_module=(
            subgraph.graph_module if has_score_mod and subgraph is not None else None
        ),
        score_mod_other_buffers=score_mod_other_buffers,
        has_mask_mod=needs_block_mask,
        has_mask_aux_tensors=len(mask_mod_other_buffers) > 0,
        mask_mod_graph_module=mask_graph.graph_module,
        mask_mod_other_buffers=mask_mod_other_buffers,
    )
    packed_mask_intervals = None
    if needs_block_mask and torch.cuda.is_available():
        device_index = None if device is None else device.index
        cuda_major = torch.cuda.get_device_capability(device_index)[0]
        if cuda_major in (10, 11):
            packed_mask_intervals = select_packed_mask_intervals(
                mask_graph.graph_module
            )
    if packed_mask_intervals is not None:
        configs = [
            FlexFlashConfig(conf.score_mod_vec_size, DEFAULT_MASK_MOD_VEC_SIZE)
            for conf in configs
        ]
        max_configs = torch._inductor.config.test_configs.max_flex_configs
        if max_configs is not None and len(configs) > max_configs:
            configs = configs[:max_configs]

    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[lse],
                subgraphs=subgraphs,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                MASK_MOD_VEC_SIZE=conf.mask_mod_vec_size,
                MASK_MOD_PACKED_INTERVALS=(
                    packed_mask_intervals if conf.mask_mod_vec_size == 32 else None
                ),
                MASK_MOD_OTHER_BUFFERS=mask_mod_other_buffers,
                NEEDS_BLOCK_MASK=needs_block_mask,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if has_full_blocks:
        input_gen_fns = {
            4: create_num_blocks_fake_generator(kv_indices),
            5: create_indices_fake,
            6: create_num_blocks_fake_generator(full_kv_indices),
            7: create_indices_fake,
        }

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, lse)


def _can_use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    num_score_mod_placeholders: int = 5,
) -> tuple[bool, str]:
    if not ensure_flash_available():
        return False, "CUTE flash attention is not available"

    if input_buffers_require_grads(
        fw_subgraph.graph_module, num_score_mod_placeholders
    ):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention backward)",
        )

    if joint_outputs is not None:
        if joint_outputs.captured_grads_compute:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support captured grads yet.",
            )
        if joint_outputs.mutated_grads:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support mutated grads yet.",
            )

    return True, ""


def _use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
) -> bool:
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        fw_subgraph: The forward score modification subgraph
        mask_graph: The mask modification subgraph
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)
        joint_outputs: Processed joint outputs (for PR1 constraint checking)
        score_mod_other_buffers: Additional buffers used by score_mod

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention_backward(
        fw_subgraph,
        mask_graph,
        joint_outputs,
        score_mod_other_buffers,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    scale: float,
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    fw_subgraph_buffer: SubgraphResults | None = None,
    joint_subgraph_buffer: Any | None = None,
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_graph_buffer: SubgraphResults | None = None,
    q_num_blocks: TensorBox | None = None,
    q_indices: TensorBox | None = None,
    full_q_num_blocks: TensorBox | None = None,
    full_q_indices: TensorBox | None = None,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    """Create a CuteDSL flash attention backward kernel for the default mod path."""
    if not ensure_flash_available():
        raise RuntimeError("CUTE flash attention not available")

    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
    device = query.get_device()
    dtype = query.get_dtype()
    assert device is not None

    grad_query_strides = infer_dense_strides(
        [batch_size, num_heads, seq_len_q, head_dim], query.get_stride()
    )
    grad_query = empty_strided(
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=grad_query_strides,
        dtype=dtype,
        device=device,
    )

    grad_key_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, head_dim], key.get_stride()
    )
    grad_key = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, head_dim],
        stride=grad_key_strides,
        dtype=dtype,
        device=device,
    )

    grad_value_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, v_head_dim], value.get_stride()
    )
    grad_value = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, v_head_dim],
        stride=grad_value_strides,
        dtype=dtype,
        device=device,
    )

    # we use dq as the output layout
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    choices: list[Any] = []

    input_nodes: list[TensorBox] = [
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        grad_key,
        grad_value,
    ]

    has_block_mask = mask_graph_buffer is not None
    if has_block_mask:
        assert q_indices is not None
        assert full_q_num_blocks is not None
        assert full_q_indices is not None
        input_nodes.extend(
            [
                cast(TensorBox, q_num_blocks),
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

    has_score_mod = fw_subgraph_buffer is not None and joint_subgraph_buffer is not None
    subgraphs = []
    if has_score_mod:
        subgraphs.append(fw_subgraph_buffer)
        subgraphs.append(joint_subgraph_buffer)
    if has_block_mask:
        subgraphs.append(mask_graph_buffer)

    configs = _get_flex_flash_bwd_configs()

    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_backward_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[grad_key, grad_value],
                subgraphs=subgraphs or None,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                HAS_BLOCK_MASK=has_block_mask,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if has_block_mask:
        input_gen_fns = {
            8: create_num_blocks_fake_generator(q_indices),
            9: create_indices_fake,
            10: create_num_blocks_fake_generator(full_q_indices),
            11: create_indices_fake,
        }

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, tuple())
