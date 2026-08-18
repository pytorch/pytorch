# mypy: allow-untyped-defs
"""Call into FlyDSL for flex attention backward (gfx950).

Mirrors the CuteDSL/FLASH backward path but targets the FlyDSL template infra.
The template body calls the vendored MFMA kernels in
``vendored_templates/flydsl/kernels/flex_attn_bwd_gfx950.py`` (delta, prologue,
dkdv, dq). Everything here is inert unless the FlyDSL runtime is installed and
the device is ROCm/gfx950. Identity score_mod only.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import sympy

import torch

from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    runtime_available,
)
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
from .flex_flash_attention import input_buffers_require_grads, is_trivial_score_graph


flex_flydsl_backward_template = FlyDSLTemplate(
    name="flex_flydsl_backward", source=load_flex_template("flydsl_backward")
)


def _flydsl_unavailable_message() -> str:
    reason = _flydsl_runtime_unavailable_reason()
    if reason is None:
        reason = "FlyDSL runtime is unavailable"
    return (
        f"FlyDSL flex attention backward is unavailable: {reason}. "
        "It requires ROCm/gfx950 and the optional `flydsl` runtime (0.3.x)."
    )


def _can_use_flydsl_flex_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    query: TensorBox,
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    num_score_mod_placeholders: int = 5,
) -> tuple[bool, str]:
    """Check if FlyDSL flex attention backward can be used for the given inputs.

    Returns (can_use, reason). Never raises: any failure returns False with a
    reason so the caller falls through to the Triton backend.
    """
    if not runtime_available():
        return False, _flydsl_unavailable_message()

    if torch.version.hip is None:
        return False, "FlyDSL flex bwd requires ROCm/gfx950"

    if query.get_dtype() != torch.bfloat16:
        return (
            False,
            f"FlyDSL flex bwd supports bf16 only, got {query.get_dtype()}",
        )

    if not is_trivial_score_graph(fw_subgraph.graph_module):
        return False, "FlyDSL flex bwd supports identity score_mod only"

    if input_buffers_require_grads(
        fw_subgraph.graph_module, num_score_mod_placeholders
    ):
        return False, "NYI: FlyDSL flex bwd doesn't support captured grads yet."

    if joint_outputs is not None:
        if joint_outputs.captured_grads_compute:
            return False, "NYI: FlyDSL flex bwd doesn't support captured grads yet."
        if joint_outputs.mutated_grads:
            return False, "NYI: FlyDSL flex bwd doesn't support mutated grads yet."

    return True, ""


def _use_flydsl_flex_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    backend: str,
    query: TensorBox,
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
) -> bool:
    """Determine if we should use FlyDSL flex attention backward.

    FlyDSL is experimental and must be explicitly requested via BACKEND='FLYDSL'.
    Mirrors ``_use_flex_flash_attention_backward``: raises when the backend is
    requested but cannot be satisfied.
    """
    if backend != "FLYDSL":
        return False

    can_use, reason = _can_use_flydsl_flex_attention_backward(
        fw_subgraph,
        mask_graph,
        query,
        joint_outputs,
        score_mod_other_buffers,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLYDSL' but FlyDSL flex backward cannot be used: {reason}"
        )

    return True


def create_flydsl_flex_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    grad_logsumexp: TensorBox | None,
    scale: float,
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    fw_subgraph_buffer: SubgraphResults | None = None,
    joint_subgraph_buffer: Any | None = None,
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_graph_buffer: SubgraphResults | None = None,
    mask_mod_other_buffers: list[TensorBox] | None = None,
    q_num_blocks: TensorBox | None = None,
    q_indices: TensorBox | None = None,
    full_q_num_blocks: TensorBox | None = None,
    full_q_indices: TensorBox | None = None,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    """Create a FlyDSL flex attention backward kernel for the default mod path.

    TODO(follow-up): FlyDSL bwd currently drops the FA4-specific dq_write_order /
    dq_kv_order / aux_scalar / deterministic machinery. Identity score_mod only.
    """
    if not runtime_available():
        raise RuntimeError(_flydsl_unavailable_message())

    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must not be None")

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
        if q_indices is None:
            raise AssertionError("q_indices required when block mask is present")
        if full_q_num_blocks is None:
            raise AssertionError("full_q_num_blocks required when block mask is present")
        if full_q_indices is None:
            raise AssertionError("full_q_indices required when block mask is present")
        input_nodes.extend(
            [
                cast(TensorBox, q_num_blocks),
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

    choices: list[Any] = []
    error = flex_flydsl_backward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=output_layout,
        mutated_inputs=[grad_key, grad_value],
        SM_SCALE=scale,
        HAS_BLOCK_MASK=has_block_mask,
        SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
        SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
    )

    if not choices:
        raise RuntimeError(f"FlyDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if has_block_mask:
        input_gen_fns = {
            8: create_num_blocks_fake_generator(q_indices),
            9: create_indices_fake,
            10: create_num_blocks_fake_generator(full_q_indices),
            11: create_indices_fake,
        }

    template_output, _ = autotune_select_algorithm(
        "flex_flydsl_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, tuple())
