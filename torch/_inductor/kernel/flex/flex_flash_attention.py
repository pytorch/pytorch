# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import torch

from ...ir import FallbackKernel, TensorBox


aten = torch.ops.aten
prims = torch.ops.prims

try:
    from flash_attn.cute import flash_attn_func  # type: ignore[import-not-found]

    CUTE_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    CUTE_AVAILABLE = False


def _use_flex_flash_attention(subgraph, mask_graph):
    """Determine if we can use flex flash attention for the given inputs."""
    if not CUTE_AVAILABLE:
        return False
    # TODO: essentially fully dense attention
    if subgraph.graph is None and mask_graph.graph is None:
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
    query,
    key,
    value,
    block_mask,
    scale,
    kernel_options,
    subgraph_buffer,
    mask_graph_buffer,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
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
