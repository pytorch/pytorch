# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import dataclasses
import functools
import importlib
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, cast, Literal

import sympy
from sympy import Expr, Integer

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import autotune_select_algorithm
from ...sizevars import stride_at
from ...virtualized import V
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    SubgraphResults,
)


@dataclasses.dataclass
class FlexFlashConfig:
    """Autotuning configuration for CuteDSL flex flash attention kernels.

    score_mod_vec_size: Number of elements processed per thread in the score_mod
        application loop. Maps to score_mod.__vec_size__ in CuTe flash attention.
        None uses the kernel default. Only effective for forward; backward does
        not currently support vectorized score_mod.
    """

    score_mod_vec_size: int | None = None


def _get_flex_flash_fwd_configs(
    has_score_mod: bool,
    has_aux_tensors: bool,
    device: torch.device | None = None,
    score_mod_graph_module: GraphModule | None = None,
    score_mod_other_buffers: Sequence[TensorBox] = (),
) -> list[FlexFlashConfig]:
    if not has_score_mod:
        return [FlexFlashConfig()]
    if has_aux_tensors:
        device_index = None if device is None else device.index
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(device_index)[0] >= 10
        ):
            return [
                FlexFlashConfig(
                    score_mod_vec_size=_select_aux_score_mod_vec_size(
                        score_mod_graph_module, score_mod_other_buffers
                    )
                )
            ]
        return [FlexFlashConfig(score_mod_vec_size=1)]
    if not torch._inductor.config.max_autotune:
        return [FlexFlashConfig()]
    return [
        FlexFlashConfig(score_mod_vec_size=v) for v in (1, 2, 4, 8, 16, 32, 64, 128)
    ]


def _select_aux_score_mod_vec_size(
    graph_module: GraphModule | None, score_mod_other_buffers: Sequence[TensorBox]
) -> int:
    """Choose a safe score_mod vector width for captured tensor loads.

    Flex score_mod vectorization is only enabled when every captured tensor
    index load can be emitted without per-lane gather semantics: either as a
    direct contiguous vector load or as a lane-uniform scalar load broadcast to
    all lanes. If any load needs scalar gather semantics, force vec_size=1 so
    the generated score_mod matches scalar-lane lowering.
    """
    if graph_module is None:
        return 1

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    if len(placeholders) < 5:
        return 1

    capture_to_buffer = dict(zip(placeholders[5:], score_mod_other_buffers))
    selected_vec_size = 8
    found_vectorizable_load = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.index.Tensor:
            continue
        buffer_node, indices = node.args
        if buffer_node not in capture_to_buffer:
            continue
        max_vec_size = _max_direct_aux_load_vec_size(
            indices, capture_to_buffer[buffer_node], placeholders[3], placeholders[4]
        )
        if max_vec_size is None:
            return 1
        selected_vec_size = min(selected_vec_size, max_vec_size)
        found_vectorizable_load = True

    return selected_vec_size if found_vectorizable_load else 1


def _max_direct_aux_load_vec_size(
    indices: object,
    buffer: TensorBox,
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
) -> int | None:
    if not isinstance(indices, (list, tuple)) or not indices:
        return None

    q_idx = sympy.Symbol("q_idx", integer=True, nonnegative=True)
    kv_idx = sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    prefix_exprs = [
        _fx_aux_index_to_sympy(index, q_idx_node, kv_idx_node, q_idx, kv_idx)
        for index in indices[:-1]
    ]
    if any(expr is None or kv_idx in expr.free_symbols for expr in prefix_exprs):
        return None

    last_expr = _fx_aux_index_to_sympy(
        indices[-1], q_idx_node, kv_idx_node, q_idx, kv_idx
    )
    if last_expr is None:
        return None
    if kv_idx not in last_expr.free_symbols:
        # All score_mod vector lanes read the same element, so the load can use
        # the uniform-broadcast path even though it is not a direct vector copy.
        return 8

    sizes = buffer.get_size()
    strides = buffer.get_stride()
    if not V.graph.sizevars.statically_known_equals(strides[-1], 1):
        return None

    # FlashAttention's score_mod loop groups consecutive flattened score entries.
    # On SM100, those entries have consecutive KV coordinates for a fixed Q row.
    offset = buffer.get_layout().offset
    lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(last_expr, kv_idx)
    for vec_size in (8, 4, 2):
        if not (
            V.graph.sizevars.statically_known_multiple_of(sizes[-1], vec_size)
            and V.graph.sizevars.statically_known_multiple_of(offset, vec_size)
            and all(
                V.graph.sizevars.statically_known_multiple_of(stride, vec_size)
                for stride in strides[:-1]
            )
        ):
            continue
        if (
            (
                isinstance(last_expr, ModularIndexing)
                or V.graph.sizevars.statically_known_equals(
                    stride_at(last_expr, kv_idx), 1
                )
            )
            and lane_contiguity.is_contiguous_for(vec_size)
            and _lane_group_start_is_aligned(last_expr, kv_idx, vec_size)
            and _lane_group_start_is_nonnegative(last_expr, kv_idx)
        ):
            return vec_size
    return None


def _fx_aux_index_to_sympy(
    index: object,
    q_idx_node: torch.fx.Node,
    kv_idx_node: torch.fx.Node,
    q_idx: sympy.Symbol,
    kv_idx: sympy.Symbol,
) -> sympy.Expr | None:
    if isinstance(index, int | sympy.Integer):
        return sympy.Integer(index)
    if not isinstance(index, torch.fx.Node):
        return None
    if index is q_idx_node:
        return q_idx
    if index is kv_idx_node:
        return kv_idx
    if index.op != "call_function":
        return None

    args = index.args
    target = index.target
    if len(args) < 2:
        return None
    lhs = _fx_aux_index_to_sympy(args[0], q_idx_node, kv_idx_node, q_idx, kv_idx)
    rhs = _fx_aux_index_to_sympy(args[1], q_idx_node, kv_idx_node, q_idx, kv_idx)
    if lhs is None or rhs is None:
        return None
    if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
        return V.graph.sizevars.simplify(lhs + rhs)
    if target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
        return V.graph.sizevars.simplify(lhs - rhs)
    if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
        return V.graph.sizevars.simplify(lhs * rhs)
    if target in (torch.ops.aten.remainder.Tensor, torch.ops.aten.remainder.Scalar):
        return ModularIndexing(lhs, 1, rhs)
    if (
        target == torch.ops.aten.div.Tensor_mode
        and index.kwargs.get("rounding_mode") == "floor"
    ):
        return FloorDiv(lhs, rhs)
    return None


def _lane_group_start_is_aligned(
    expr: sympy.Expr, lane_var: sympy.Symbol, vec_size: int
) -> bool:
    start_expr = V.graph.sizevars.simplify(expr.xreplace({lane_var: sympy.Integer(0)}))
    return V.graph.sizevars.statically_known_multiple_of(start_expr, vec_size)


def _lane_group_start_is_nonnegative(expr: sympy.Expr, lane_var: sympy.Symbol) -> bool:
    start_expr = V.graph.sizevars.simplify(expr.xreplace({lane_var: sympy.Integer(0)}))
    return V.graph.sizevars.statically_known_geq(start_expr, 0)


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


@functools.lru_cache(maxsize=1)
def _is_symbol_from_tensor_shape(symbol: sympy.Symbol, shape_env: Any) -> bool:
    """Check if a symbol originates from a tensor size/stride (TensorPropertySource)."""
    from torch._dynamo.source import TensorPropertySource

    sources = shape_env.var_to_sources.get(symbol, [])
    return any(isinstance(s, TensorPropertySource) for s in sources)


def _has_unsupported_captured_scalars(
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> bool:
    """Check if any captured buffers are dynamic scalars that cannot be inlined.

    When compiling with dynamic=True, captured Python scalars in score_mod or
    mask_mod may become:
    - sympy symbols from LocalSource (captured ints) - NOT from tensor shapes
    - 0-dim CPU tensors (captured floats)

    Symbols from TensorPropertySource (tensor size/stride) are fine because they
    get resolved at runtime.

    The FLASH backend cannot inline captured scalar symbolic values into the CuteDSL template.
    """
    from torch._inductor.virtualized import V

    shape_env = V.graph.sizevars.shape_env

    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        # Captured int becomes sympy.Symbol - check if it's NOT from a tensor shape
        if isinstance(buf, sympy.Expr):
            for symbol in buf.free_symbols:
                if not _is_symbol_from_tensor_shape(symbol, shape_env):
                    return True
        # Captured float becomes 0-dim TensorBox on CPU
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and device.type == "cpu" and len(size) == 0:
                # 0-dimensional CPU tensor (scalar) - can't be inlined into CUDA kernel
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

    configs = _get_flex_flash_fwd_configs(
        has_score_mod,
        len(score_mod_other_buffers) > 0,
        device,
        subgraph.graph_module if has_score_mod and subgraph is not None else None,
        score_mod_other_buffers,
    )

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
