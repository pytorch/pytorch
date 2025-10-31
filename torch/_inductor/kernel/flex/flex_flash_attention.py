# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import functools
import importlib
from typing import Any

import sympy

import torch
from torch.fx import GraphModule

from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from .common import infer_dense_strides, load_flex_template, SubgraphResults


aten = torch.ops.aten
prims = torch.ops.prims


@functools.lru_cache(maxsize=1)
def ensure_flash_available() -> bool:
    """Check if flash-attn is importable; cache the result for reuse.

    Call ensure_flash_available.cache_clear() after installing flash-attn
    in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except ImportError:
        return False


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_flex_template("flash_attention")
)


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


def is_trivial_graph(
    graph_module: GraphModule, is_score_graph: bool, num_score_mod_placeholders: int
):
    """Check if the flex graphs are compatible with Flash Attention."""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    assert len(output) == 1, "Got graph w/ multiple outputs"
    output_val = output[0].args[0]

    if is_score_graph:
        if input_buffers_require_grads(graph_module, num_score_mod_placeholders):
            return False
        return True  # party on garth
    # mask mod graph is empty if we have 4 inputs and full_default output
    return len(placeholders) == 4 and output_val.target == torch.ops.aten.full.default


def _can_use_flex_flash_attention(
    subgraph: Subgraph, mask_graph: Subgraph, num_score_mod_placeholders: int
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

    score_trivial = is_trivial_graph(
        subgraph.graph_module,
        is_score_graph=True,
        num_score_mod_placeholders=num_score_mod_placeholders,
    )
    mask_trivial = is_trivial_graph(
        mask_graph.graph_module,
        is_score_graph=False,
        num_score_mod_placeholders=num_score_mod_placeholders,
    )

    if not score_trivial and not mask_trivial:
        return (
            False,
            "Both score and mask graphs are too complex for flash attention (require simple operations only)",
        )
    elif not score_trivial:
        return (
            False,
            "Score modification captured tensors that require gradients (not supported by flash attention)",
        )
    elif not mask_trivial:
        return (
            False,
            "A non None BlockMask was passed to flex attention (not supported by flash attention yet)",
        )

    return True, ""


def _use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    kernel_options: dict[str, Any],
    num_score_mod_placeholders: int,
) -> bool:
    """Determine if we should use flex flash attention for the given inputs."""
    force_flash = kernel_options.get("force_flash", False)

    can_use, reason = _can_use_flex_flash_attention(
        subgraph, mask_graph, num_score_mod_placeholders
    )

    if force_flash and not can_use:
        raise RuntimeError(
            f"force_flash=True but flash attention cannot be used: {reason}"
        )

    return force_flash and can_use


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
