# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

from typing import Any

import torch
from torch.fx import GraphModule

from ...ir import FallbackKernel, ShapeAsConstantBuffer, Subgraph, TensorBox
from .common import SubgraphResults


aten = torch.ops.aten
prims = torch.ops.prims

try:
    from flash_attn.cute import flash_attn_func  # type: ignore[import-not-found]

    CUTE_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    CUTE_AVAILABLE = False


def is_trivial_graph(graph_module: GraphModule, is_score_graph: bool):
    """Check if the flex graphs are trivial"""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    # Check if it's just placeholder -> output
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    assert len(output) == 1, "Got graph w/ multiple outputs"
    output_val = output[0].args[0]
    if is_score_graph:
        return len(placeholders) == 5 and output_val == placeholders[0]
    # mask mod graph is empty if we have 4 inputs and full_default output
    return len(placeholders) == 4 and output_val.target == torch.ops.aten.full.default


def _use_flex_flash_attention(
    subgraph: Subgraph, mask_graph: Subgraph, kernel_options: dict[str, Any]
) -> bool:
    """Determine if we can use flex flash attention for the given inputs."""
    if not CUTE_AVAILABLE:
        return False
    if kernel_options.get("disable_flash", False):
        return False
    if is_trivial_graph(subgraph.graph_module, True) and is_trivial_graph(
        mask_graph.graph_module, False
    ):
        return True

    return False


@torch.library.custom_op("flex_flash_attn::flash_attn_fwd", mutates_args=())
def flash_attention_forward_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal flash attention forward kernel using CUTE implementation."""
    if not CUTE_AVAILABLE:
        raise RuntimeError("CUTE flash attention not available")
    assert flash_attn_func is not None

    q_transposed = query.transpose(1, 2)
    k_transposed = key.transpose(1, 2)
    v_transposed = value.transpose(1, 2)

    output, lse = flash_attn_func(
        q_transposed,
        k_transposed,
        v_transposed,
        softmax_scale=scale,
        causal=causal,
    )

    return output.transpose(1, 2), lse


@torch.library.register_fake("flex_flash_attn::flash_attn_fwd")  # type: ignore[misc]
def flex_flash_attn_fwd_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for the custom op."""
    batch_size, num_heads, seqlen_q, head_dim = query.shape

    out = query.new_empty(batch_size, seqlen_q, num_heads, head_dim).transpose(1, 2)
    lse = query.new_empty(batch_size, num_heads, seqlen_q, dtype=torch.float32)

    return out, lse


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
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox | ShapeAsConstantBuffer]:
    """Create a flex flash attention kernel."""
    if not CUTE_AVAILABLE:
        raise RuntimeError("CUTE flash attention not available")

    outputs = FallbackKernel.create(
        torch.ops.flex_flash_attn.flash_attn_fwd.default,
        query,
        key,
        value,
        scale=scale,
        causal=False,
    )
    assert isinstance(outputs, (tuple, list))
    return TensorBox.create(outputs[0]), TensorBox.create(outputs[1])
