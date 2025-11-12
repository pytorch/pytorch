# mypy: allow-untyped-defs
"""Call into flash-attention 4 for flexattention"""

import functools
import importlib
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Optional

import sympy
from sympy import Expr, Integer

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import Identity

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
        return importlib.util.find_spec("flash_attn.cute") is not None  # type: ignore[attr-defined]
    except ImportError:
        return False


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_flex_template("flash_attention")
)


def _fixed_indexer_cute(
    size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[Expr]], Expr]:
    """
    Colexicographic indexer for CuteDSL - matches CuTe's coordinate interpretation.

    CuTe interprets linear indices in colexicographic (column-major) order,
    whereas Inductor's default _fixed_indexer uses lexicographic (row-major) order.

    For size=[4, 128] with index=[b, q_idx]:
    - Lexicographic:    b*128 + q_idx*1
    - Colexicographic:  b*1 + q_idx*2

    CuTe then applies the tensor's actual memory strides to get the correct offset.
    """

    def indexer(index: Sequence[Expr]) -> Expr:
        assert offset == Integer(0), "Offset not supported for colexicographic indexing"
        if not index:
            return Integer(0)

        result = index[0]
        runner = size[0]

        for idx, sz in zip(index[1:], size[1:], strict=True):
            result = result + runner * Identity(idx)
            runner = runner * sz

        return result

    return indexer


@contextmanager
def patch_fixed_layout_indexer_for_cutedsl():
    """
    Temporarily swap FixedLayout.make_indexer so CuteDSL sees colexicographic indexing.

    Note [CuteDSL indexer patch]:
    Flex flash attention only supports a limited set of IR ops (pointwise, reads, no stores),
    so temporarily changing the indexing order is safe for the kernels we emit today.
    TODO(dynamic shapes): Reconfirm once flex flash attention supports dynamic shapes.
    """
    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self):
        return _fixed_indexer_cute(self.size, self.stride, self.offset)

    FixedLayout.make_indexer = cutedsl_make_indexer  # type: ignore[assignment]
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer  # type: ignore[assignment]


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
def _supports_nontrivial_mask_graphs() -> bool:
    """Currently only supported on Hopper (SM90) GPUs."""
    return torch.cuda.get_device_capability()[0] == 9


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
    mask_trivial = is_trivial_mask_graph(mask_graph.graph_module)

    if mask_trivial:
        return True, ""

    if not _supports_nontrivial_mask_graphs():
        return (
            False,
            "NYI: Non-trivial mask graphs only supported on Hopper (SM90) for flash attention",
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
    kv_num_blocks: TensorBox | None,
    kv_indices: TensorBox | None,
    full_kv_num_blocks: TensorBox | None,
    full_kv_indices: TensorBox | None,
    mask_graph: Subgraph,
    subgraph: Subgraph | None = None,
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

    # Used to check if we can skip block sparse impl
    mask_graph_is_trivial = is_trivial_mask_graph(mask_graph.graph_module)

    needs_block_mask = not mask_graph_is_trivial
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

    error = flash_attention_cutedsl_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=output_layout,
        mutated_inputs=[lse],
        subgraphs=[subgraph_buffer, mask_graph_buffer],
        SM_SCALE=scale,
        NEEDS_BLOCK_MASK=needs_block_mask,
    )

    def wrap_choice_render(choice):
        # See Note [CuteDSL indexer patch]
        original_make_kernel_render = choice.make_kernel_render

        def make_kernel_render_with_patch(*args, **kwargs):
            render_kernel, render = original_make_kernel_render(*args, **kwargs)

            # Let the template construct its closures, then scope the indexer patch
            # to the actual render call that emits the kernel
            render_with_patch = patch_fixed_layout_indexer_for_cutedsl()(render)

            return render_kernel, render_with_patch

        choice.make_kernel_render = make_kernel_render_with_patch

    for choice in choices:
        wrap_choice_render(choice)

    if error or not choices:
        # Fallback to original implementation
        raise RuntimeError(f"CuteDSL template failed: {error}")

    # No autotune for now
    template_output = choices[0].output_node()

    return (template_output, lse)
