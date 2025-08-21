# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

from typing import Any

import sympy

import torch
from torch.fx import GraphModule

from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from .common import infer_dense_strides, load_template, SubgraphResults


aten = torch.ops.aten
prims = torch.ops.prims

try:
    from flash_attn.cute import flash_attn_func  # type: ignore[import-not-found]

    CUTE_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    CUTE_AVAILABLE = False


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_template("flash_attention")
)


def is_trivial_graph(graph_module: GraphModule, is_score_graph: bool):
    """Check if the flex graphs are trivial"""
    return True
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
    """Create a flex flash attention kernel using CuteDSL template."""
    if not CUTE_AVAILABLE:
        raise RuntimeError("CUTE flash attention not available")

    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()

    # Ensure device is not None
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

    choices: list[Any] = []
    causal = kernel_options.get("causal", False)

    assert flash_attention_cutedsl_template is not None
    error = flash_attention_cutedsl_template.maybe_append_choice(
        choices,
        input_nodes=[query, key, value, lse],
        layout=output_layout,
        mutated_inputs=[lse],
        subgraphs=[subgraph_buffer, mask_graph_buffer],
        SM_SCALE=scale,
        CAUSAL=causal,
    )

    if error or not choices:
        # Fallback to original implementation
        raise RuntimeError(f"CuteDSL template failed: {error}")

    # No autotune for now
    template_output = choices[0].output_node()

    return (template_output, lse)
