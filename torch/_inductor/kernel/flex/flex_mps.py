# mypy: allow-untyped-defs
"""MPS-specific lowering for flex attention."""

import logging

import sympy

import torch
from torch._inductor.virtualized import V

from ...ir import FixedLayout, TensorBox
from ...lowering import _full
from ...select_algorithm import realize_inputs
from .common import (
    infer_dense_strides,
    maybe_realize,
)


log = logging.getLogger(__name__)


def lower_mps(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    """Metal-based flex attention for MPS devices.

    Generates a Metal shader implementing online-softmax attention
    with block sparsity and inlined score_mod / mask_mod.
    """
    from ...codegen.metal_flex_attention_template import (
        _generate_metal_shader,
        MetalFlexAttentionNode,
    )

    # Unpack block mask
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    # Validate dtypes
    if query.get_dtype() not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"flex_attention on MPS supports float32/float16/bfloat16, got {query.get_dtype()}"
        )
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        raise NotImplementedError(
            "Mixed dtypes for query, key, value not supported on MPS"
        )

    if kernel_options.get("OUTPUT_LOGSUMEXP", False):
        raise NotImplementedError(
            "flex_attention backward (return_lse=True) is not yet supported on MPS. "
            "Use torch.no_grad() or torch.inference_mode() for inference."
        )

    dtype = query.get_dtype()

    # Realize inputs
    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [query, key, value, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices]
    )

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    B = Bq

    has_full_blocks = full_kv_num_blocks is not None

    # Guard compile-time constants
    d_qk = V.graph.sizevars.guard_int(qk_head_dim)
    d_v = V.graph.sizevars.guard_int(v_head_dim)
    SPARSE_KV_BLOCK_SIZE_val = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE_val = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)

    # GQA factor, TODO do we need checking here for division of Hq / Hkv?
    enable_gqa = V.graph.sizevars.evaluate_expr(sympy.Ne(Hq, Hkv))
    gqa_shared_heads = sympy.floor(Hq / Hkv) if enable_gqa else sympy.Integer(1)

    BLOCK_M = 32

    # Generate Metal shader with compile-time constants and inlined score/mask mods
    shader_source = _generate_metal_shader(
        dtype=dtype,
        d_qk=d_qk,
        d_v=d_v,
        score_mod_graph=subgraph.graph_module,
        mask_mod_graph=mask_graph.graph_module,
        num_extra_bufs=0,
        has_full_blocks=has_full_blocks,
        block_m=BLOCK_M,
    )

    # Output layout matching query strides
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())
    layout = FixedLayout(
        query.get_device(),
        dtype,
        out_size,
        stride=[sympy.sympify(s) for s in out_strides],
    )

    # Create scale as a 1-element float32 tensor (IR-level), TODO is this needed? like the isinstance thingie
    scale_val = scale if isinstance(scale, (int, float)) else float(scale)
    scale_ir = _full(scale_val, query.get_device(), torch.float32, [1])
    scale_ir.realize()

    # Get sparse block strides before realize_inputs (which may wrap nodes)
    kv_nb_strides = _get_strides(kv_num_blocks)   # [stride_z, stride_h, stride_q]
    kv_idx_strides = _get_strides(kv_indices)       # [stride_z, stride_h, stride_q, stride_b]
    full_kv_nb_strides = _get_strides(full_kv_num_blocks) if has_full_blocks else []
    full_kv_idx_strides = _get_strides(full_kv_indices) if has_full_blocks else []

    # Build tensor inputs (must match shader [[buffer(N)]] order)
    # Buffer 0 = Out, 1=Q, 2=K, 3=V, 4=kv_num_blocks, 5=kv_indices, 6=scale
    input_nodes = [query, key, value, kv_num_blocks, kv_indices, scale_ir]
    if has_full_blocks:
        input_nodes += [full_kv_num_blocks, full_kv_indices]

    realized_inputs = realize_inputs(*input_nodes)

    # Get sparse structure sizes (SPARSE_Z, SPARSE_HQ) for modular indexing
    kv_nb_sizes = kv_num_blocks.get_size()
    sparse_z = kv_nb_sizes[0] if len(kv_nb_sizes) >= 1 else sympy.Integer(1)
    sparse_hq = kv_nb_sizes[1] if len(kv_nb_sizes) >= 2 else sympy.Integer(1)

    # Build scalar args (sizes, strides) - these become constant long& in the shader
    # Order must match the scalar parameter declarations in the shader
    scalar_args = [
        B, Hq, Hkv, seq_len_q, seq_len_kv,
        # Q strides (4)
        *query.get_stride(),
        # K strides (4)
        *key.get_stride(),
        # V strides (4)
        *value.get_stride(),
        # Output strides (4)
        *[sympy.sympify(s) for s in out_strides],
        # Block mask params
        sympy.Integer(SPARSE_KV_BLOCK_SIZE_val),
        gqa_shared_heads,
        # Sparse structure dimensions for modular indexing
        sparse_z,
        sparse_hq,
        # kv_num_blocks strides [z, h, q]
        *_pad_strides(kv_nb_strides, 3),
        # kv_indices strides [z, h, q, b(lock)]
        *_pad_strides(kv_idx_strides, 4),
        # SPARSE_Q_BLOCK_SIZE
        sympy.Integer(SPARSE_Q_BLOCK_SIZE_val),
    ]
    if has_full_blocks:
        scalar_args += [
            *_pad_strides(full_kv_nb_strides, 3),
            *_pad_strides(full_kv_idx_strides, 4),
        ]

    # Grid: (ceil(N_Q / BLOCK_M), Hq, B)
    # TODO what is this
    grid = (
        sympy.ceiling(seq_len_q / BLOCK_M),
        Hq,
        B,
    )

    node = MetalFlexAttentionNode(
        layout=layout,
        inputs=realized_inputs,
        shader_source=shader_source,
        scalar_args=scalar_args,
        grid=grid,
        block_m=BLOCK_M,
    )

    return (TensorBox.create(node),)


def _get_strides(node):
    """Safely get strides from an IR node, returning empty list for None."""
    if node is None:
        return []
    try:
        return list(node.get_stride())
    except Exception:
        return []


def _pad_strides(strides, target_len):
    """Pad a strides list to target_len with Integer(1) for missing dims."""
    result = list(strides[:target_len])
    while len(result) < target_len:
        result.append(sympy.Integer(1))
    return result
