# mypy: allow-untyped-defs
"""MPS-specific lowering for flex attention."""

import sympy

import torch
from torch._inductor.virtualized import V

from ...ir import FixedLayout, TensorBox
from ...select_algorithm import realize_inputs
from .common import infer_dense_strides, maybe_realize


BLOCK_M = 32


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
    """Lower flex_attention to a Metal shader for MPS."""
    from ...codegen.metal_flex_attention_template import (
        _generate_metal_shader,
        MetalFlexAttentionNode,
    )

    if score_mod_other_buffers or mask_mod_other_buffers:
        raise NotImplementedError(
            "flex_attention on MPS does not yet support score_mod / mask_mod "
            "with captured buffers"
        )

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks
        _,  # full_q_indices
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

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
    if kernel_options.get("OUTPUT_MAX", False):
        raise NotImplementedError(
            "flex_attention on MPS does not yet support returning max scores "
            "(return_aux=AuxRequest(max_scores=True))."
        )

    dtype = query.get_dtype()

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )

    B, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    _, _, _, k_head_dim = key.get_size()

    has_full_blocks = full_kv_num_blocks is not None

    d_qk = V.graph.sizevars.guard_int(qk_head_dim)
    d_v = V.graph.sizevars.guard_int(v_head_dim)
    d_k = V.graph.sizevars.guard_int(k_head_dim)
    if d_k != d_qk:
        raise NotImplementedError(
            f"flex_attention on MPS requires query and key to share head dim, "
            f"got d_qk={d_qk} d_k={d_k}"
        )

    sizevars = V.graph.sizevars
    # Kernel indexes K/V with the same b_idx as Q; no broadcast logic for Bkv=1 yet.
    # check_equals adds a runtime guard B == Bkv; it raises AssertionError if it
    # can prove they differ — convert to NIE so callers see the right error.
    try:
        sizevars.check_equals(B, Bkv)
    except AssertionError:
        raise NotImplementedError(
            f"flex_attention on MPS does not yet support batch broadcasting "
            f"between query and key/value (Bq != Bkv); got Bq={B} Bkv={Bkv}"
        ) from None

    SPARSE_KV_BLOCK_SIZE_val = sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE_val = sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    if SPARSE_Q_BLOCK_SIZE_val < BLOCK_M or SPARSE_Q_BLOCK_SIZE_val % BLOCK_M != 0:
        # Each threadgroup tiles BLOCK_M query rows and looks up sparse-mask info
        # at sparse_q_idx = m_base / SPARSE_Q_BLOCK_SIZE; a smaller/non-multiple
        # SPARSE_Q_BLOCK_SIZE makes one threadgroup span multiple sparse blocks.
        raise NotImplementedError(
            f"flex_attention on MPS requires SPARSE_Q_BLOCK_SIZE to be a positive "
            f"multiple of {BLOCK_M}, got {SPARSE_Q_BLOCK_SIZE_val}"
        )

    if not sizevars.statically_known_multiple_of(Hq, Hkv):
        raise NotImplementedError(
            f"flex_attention on MPS requires Hq to be a positive multiple of "
            f"Hkv, got Hq={Hq} Hkv={Hkv}"
        )
    gqa_shared_heads = Hq // Hkv

    scale_val = float(scale)

    shader_source = _generate_metal_shader(
        dtype=dtype,
        d_qk=d_qk,
        d_v=d_v,
        score_mod_graph=subgraph.graph_module,
        mask_mod_graph=mask_graph.graph_module,
        has_full_blocks=has_full_blocks,
        block_m=BLOCK_M,
        scale=scale_val,
    )

    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())
    layout = FixedLayout(
        query.get_device(),
        dtype,
        out_size,
        stride=[sympy.sympify(s) for s in out_strides],
    )

    kv_nb_strides = _get_strides(kv_num_blocks)
    kv_idx_strides = _get_strides(kv_indices)
    full_kv_nb_strides = _get_strides(full_kv_num_blocks) if has_full_blocks else []
    full_kv_idx_strides = _get_strides(full_kv_indices) if has_full_blocks else []

    # Buffer order: 0=Out, 1=Q, 2=K, 3=V, 4=kv_num_blocks, 5=kv_indices,
    # 6=(full_kv_num_blocks), 7=(full_kv_indices), then packed scalar buffer.
    input_nodes = [query, key, value, kv_num_blocks, kv_indices]
    if has_full_blocks:
        input_nodes += [full_kv_num_blocks, full_kv_indices]

    realized_inputs = realize_inputs(*input_nodes)

    # Left-pad to (Z, H, Q_blocks); a sliced kv_num_blocks may be 1D or 2D.
    kv_nb_sizes = list(kv_num_blocks.get_size())
    while len(kv_nb_sizes) < 3:
        kv_nb_sizes.insert(0, sympy.Integer(1))
    sparse_z = kv_nb_sizes[0]
    sparse_hq = kv_nb_sizes[1]

    # Order must match the unpack in metal_flex_attention_template's `scalar_names`.
    scalar_args = [
        B,
        Hq,
        Hkv,
        seq_len_q,
        seq_len_kv,
        *query.get_stride(),
        *key.get_stride(),
        *value.get_stride(),
        *[sympy.sympify(s) for s in out_strides],
        sympy.Integer(SPARSE_KV_BLOCK_SIZE_val),
        gqa_shared_heads,
        sparse_z,
        sparse_hq,
        *_pad_strides(kv_nb_strides, 3),
        *_pad_strides(kv_idx_strides, 4),
        sympy.Integer(SPARSE_Q_BLOCK_SIZE_val),
    ]
    if has_full_blocks:
        scalar_args += [
            *_pad_strides(full_kv_nb_strides, 3),
            *_pad_strides(full_kv_idx_strides, 4),
        ]

    # (num_q_blocks, Hq, B); each threadgroup owns BLOCK_M query rows.
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
    if node is None:
        return []
    return list(node.get_stride())


def _pad_strides(strides, target_len):
    """Left-pad strides to target_len with 0 (broadcast over missing leading dims)."""
    strides = list(strides)
    if len(strides) > target_len:
        strides = strides[-target_len:]
    while len(strides) < target_len:
        strides.insert(0, sympy.Integer(0))
    return strides
