# mypy: allow-untyped-defs
"""Main flex_attention kernel implementation"""

import logging
import math
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any

import sympy

import torch
from torch._inductor.utils import can_use_tma
from torch._inductor.virtualized import V

from ...ir import FixedLayout, TensorBox
from ...lowering import empty, empty_strided, lowerings, register_lowering
from ...select_algorithm import autotune_select_algorithm, TritonTemplate
from .common import (
    build_subgraph_buffer,
    compute_forward_block_mn,
    compute_forward_inner,
    compute_next_offset_func,
    create_indices_fake,
    create_num_blocks_fake_generator,
    create_placeholder,
    flex_attention_grid,
    get_bounded_indices_func,
    get_float32_precision,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    load_checked_block,
    maybe_realize,
    set_head_dim_values,
)
from .flex_cpu import lower_cpu
from .flex_decoding import create_flex_decoding_kernel


log = logging.getLogger(__name__)
aten = torch.ops.aten
Expr = sympy.Expr


compute_flex_attention = r"""
{{def_kernel("Q", "K", "V", "LSE", "KV_NUM_BLKS", "KV_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0)
    off_zq = tl.program_id(1)
    off_hq = tl.program_id(2)

    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    # Setting up the TMA descriptors for Q, K, V
    desc_q = None
    desc_k = None
    desc_v = None
    {%- if USE_TMA %}
    desc_q = tl.make_tensor_descriptor(
        base=Q,
        shape=[Q_LEN, QK_HEAD_DIM],
        strides=[stride_qm, 1],
        block_shape=[BLOCK_M, QK_HEAD_DIM_ROUNDED],
    )

    desc_k = tl.make_tensor_descriptor(
        base=K,
        shape=[KV_LEN, QK_HEAD_DIM],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, QK_HEAD_DIM_ROUNDED],
    )

    desc_v = tl.make_tensor_descriptor(
        base=V,
        shape=[KV_LEN, V_HEAD_DIM],
        strides=[stride_vn, 1],
        block_shape=[BLOCK_N, V_HEAD_DIM_ROUNDED],
    )
    {%- endif %}

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950
    K_block_ptr = None
    V_block_ptr = None
    Q_block_ptr = None

    if not USE_TMA:
        Q_block_ptr = tl.make_block_ptr(
            base=Q ,
            shape=(Q_LEN, QK_HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(q_start * BLOCK_M, 0),
            block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )

    {%- if USE_TMA %}
    q = tl.load_tensor_descriptor(
        desc_q,
        [(q_start * BLOCK_M).to(tl.int32), 0],
    )
    {%- else %}
        q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    {%- endif %}

    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))


    if not USE_TMA:
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )

        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )

    offs_n = kv_start + tl.arange(0, BLOCK_N)


    acc, l_i, m_i = forward_inner(
        {{gen_argdefs()}},
        q, K_block_ptr, V_block_ptr,
        desc_k, desc_v, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_start,
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        if not USE_TMA:
            K_block_ptr = tl.make_block_ptr(
                base=K,
                shape=(QK_HEAD_DIM, KV_LEN),
                strides=(stride_kk, stride_kn),
                offsets=(0, kv_start),
                block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
                order=(0, 1)
            )
            V_block_ptr = tl.make_block_ptr(
                base=V,
                shape=(KV_LEN, V_HEAD_DIM),
                strides=(stride_vn, stride_vk),
                offsets=(kv_start, 0),
                block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
                order=(1, 0)
            )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            {{gen_argdefs()}},
            q, K_block_ptr, V_block_ptr,
            desc_k, desc_v, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_start,
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1)
    idx_hq = tl.program_id(2)
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}

    if OUTPUT_LOGSUMEXP:
        off_hz = off_zq * HQ + off_hq
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
 """


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=compute_flex_attention
    + compute_forward_inner
    + compute_next_offset_func
    + compute_forward_block_mn
    + load_checked_block
    + get_bounded_indices_func,
)


def _use_flex_decoding(query, kv_indices, kernel_options, enable_gqa):
    """Decide which kernel to use, return true if use flex decoding kernel.
    Note:
       Since the number of splits is calculated based of the the number of batch and head dims
       we need to ensure that the batch and head dims are statically known. Otherwise we just
       use the main flex_attention kernel.
    """
    force_flex = kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)
    short_query_length = V.graph.sizevars.evaluate_expr(
        sympy.Lt(query.get_size()[-2], 128)
    )
    non_zero_length = V.graph.sizevars.evaluate_expr(sympy.Gt(query.get_size()[-2], 0))
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))
    if enable_gqa:
        # in the current flex decoding triton kernel, grouped query heads for the
        # same kv head are handled by the same block. So it's hard to support different
        # kv num blocks for grouped query heads. We just fall back to main flex_attention
        # kernel where each query head is handled by a separate block.
        valid_block_mask_num_heads = V.graph.sizevars.evaluate_expr(
            sympy.Eq(kv_indices.get_size()[1], 1)
        )
    else:
        valid_block_mask_num_heads = V.graph.sizevars.evaluate_expr(
            sympy.Or(
                sympy.Eq(kv_indices.get_size()[1], 1),
                sympy.Eq(kv_indices.get_size()[1], query.get_size()[1]),
            )
        )
    return (
        not force_flex
        and short_query_length
        and static_batch
        and static_num_heads
        and non_zero_length
        and valid_block_mask_num_heads
    )


class Mode(Enum):
    fwd = auto()
    bwd = auto()


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n):
    if n == 0:
        return 1
    return 2 ** (n - 1).bit_length()


@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
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
    # For now, we don't want to pad to 8 in the forward pass, but we want to
    # do so in the backwards pass (since padding in the bwd makes a bigger perf difference)
    kernel_options = dict(kernel_options)
    assert score_mod_other_buffers is not None
    assert mask_mod_other_buffers is not None

    if query.get_device().type == "cpu":
        return lower_cpu(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    # below is cuda path if device is not cpu
    assert query.get_device().type == "cuda"
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

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    kernel_options = dict(kernel_options)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }

    assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), (
        f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
    )

    B = Bq

    kernel_options.setdefault("fwd_FLOAT32_PRECISION", get_float32_precision())
    kernel_options.setdefault("bwd_FLOAT32_PRECISION", get_float32_precision())

    def convert_mask_mod_to_fill(mask_mod_subgraph):
        """converts the mask_mod subgraph into a "fill" graph which only depends on kv_idx."""
        from torch.fx.experimental.proxy_tensor import make_fx

        def joint_fill_graph(kv_idx):
            b, h, m, kv_idx_size = 0, 0, 0, kv_idx.shape[-1]

            def mask_mod(*args):
                return lowerings[mask_mod_subgraph.graph_module](*args)

            mask = mask_mod(b, h, m, kv_idx)
            return mask.to(torch.float32)

        joint_fake_mode = V.graph.current_node.fake_mode
        with V.set_fake_mode(joint_fake_mode):
            fill_graph = make_fx(joint_fill_graph, tracing_mode="fake")(
                torch.zeros(1, dtype=torch.int32, device=query.get_device())
            ).graph

        placeholder_inps = [
            create_placeholder(name, dtype, query.get_device(), size)
            for name, dtype, size in [
                ("kv_idx", torch.int32, [1]),
            ]
        ]

        from ..ir import Subgraph

        fill_subgraph = Subgraph(
            name="fill_graph",
            graph_module=fill_graph,
            graphs=[fill_graph],
            graph_fn=joint_fill_graph,
            root_graph_lowering=V.graph,
        )
        fill_graph_buffer = build_subgraph_buffer(
            placeholder_inps + list(mask_mod_other_buffers), fill_subgraph
        )[0]
        return fill_graph_buffer

    (
        QK_HEAD_DIM,
        QK_HEAD_DIM_ROUNDED,
        V_HEAD_DIM,
        V_HEAD_DIM_ROUNDED,
    ) = set_head_dim_values(query, key, value)

    assert QK_HEAD_DIM == qk_head_dim, (
        f"Internal error: {QK_HEAD_DIM=} != {qk_head_dim=}"
    )
    assert V_HEAD_DIM == v_head_dim, f"Internal error: {V_HEAD_DIM=} != {v_head_dim=}"

    kernel_options.setdefault("fwd_BLOCK_M", 128)
    kernel_options.setdefault("bwd_BLOCK_M1", 128)
    kernel_options.setdefault("bwd_BLOCK_M2", 64)
    # We make different choices for BLOCK_N for different head dimensions
    # (in the case that head dimension is not a multiple of 128)
    kernel_options.setdefault("fwd_BLOCK_N", 64 if int(QK_HEAD_DIM) < 128 else 128)
    kernel_options.setdefault("bwd_BLOCK_N1", 64 if int(QK_HEAD_DIM) < 128 else 128)
    kernel_options.setdefault("bwd_BLOCK_N2", 128)

    kernel_options.setdefault("fwd_PRESCALE_QK", False)
    kernel_options.setdefault("bwd_PRESCALE_QK", False)

    enable_gqa = V.graph.sizevars.evaluate_expr(sympy.Ne(Hq, Hkv))
    kernel_options.setdefault(
        "fwd_GQA_SHARED_HEADS", Hq // Hkv if enable_gqa else 1
    )  # Each KV head is shared among GQA_SHARED_HEADS query heads
    kernel_options.setdefault("bwd_GQA_SHARED_HEADS", Hq // Hkv if enable_gqa else 1)
    kernel_options.setdefault("fwd_SM_SCALE", scale)
    kernel_options.setdefault("bwd_SM_SCALE", scale)
    # TODO: Check with Shunting about this tma stuff
    kernel_options.setdefault("fwd_USE_TMA", can_use_tma())
    kernel_options.setdefault("bwd_USE_TMA", can_use_tma())

    # Determine if there are "full" blocks where we only need to apply score_mod, and can skip mask_mod
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("fwd_HAS_FULL_BLOCKS", has_full_blocks)
    kernel_options.setdefault("bwd_HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        # Create a plackeholder full block list in case it is empty
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )
        full_q_num_blocks, full_q_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    kernel_options.setdefault("OUTPUT_LOGSUMEXP", kernel_options["return_lse"])

    kernel_options.setdefault("IS_DIVISIBLE", seq_len_q % 128 == 0)

    kernel_options.setdefault("fwd_SAFE_HEAD_DIM", QK_HEAD_DIM == QK_HEAD_DIM_ROUNDED)
    kernel_options.setdefault("bwd_SAFE_V_HEAD_DIM", V_HEAD_DIM == V_HEAD_DIM_ROUNDED)

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    kernel_options.setdefault("fwd_SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
    kernel_options.setdefault("fwd_SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
    kernel_options.setdefault("bwd_SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
    kernel_options.setdefault("bwd_SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

    if "bwd_fill_buffer" not in kernel_options:
        kernel_options["bwd_fill_buffer"] = convert_mask_mod_to_fill(mask_graph)

    if _use_flex_decoding(query, kv_indices, kernel_options, enable_gqa):
        return create_flex_decoding_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph,
            mask_graph,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    out_size = [B, Hq, seq_len_q, V_HEAD_DIM]
    out_strides = infer_dense_strides(out_size, query.get_stride())

    fw_out = empty_strided(
        out_size,
        out_strides,
        dtype=query.get_dtype(),
        device=query.get_device(),
    )
    fw_lse = empty_strided(
        (B, Hq, seq_len_q),
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )

    choices: list[Any] = []
    configs = V.choices.get_flex_attention_configs(
        block_mask,
        QK_HEAD_DIM,
        V_HEAD_DIM,
        key.get_dtype(),
        Mode.fwd,
        query.get_size(),
    )

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    for fwd_config in configs:
        # Remove prefix for forward kernels options and delete backward kernel options.
        cur_kernel_options = kernel_options.copy()
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)

        # Performance tuning
        cur_kernel_options.setdefault("BLOCK_M", fwd_config.block_m)
        cur_kernel_options.setdefault("BLOCK_N", fwd_config.block_n)
        cur_kernel_options.setdefault("num_warps", fwd_config.num_warps)
        cur_kernel_options.setdefault("num_stages", fwd_config.num_stages)

        # Add ROCm-specific parameters if they exist in the config
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(fwd_config, attrib):
                cur_kernel_options[attrib] = getattr(fwd_config, attrib)

        flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                fw_lse,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            mutated_inputs=[fw_lse],
            layout=FixedLayout(
                fw_out.get_device(),
                fw_out.get_dtype(),
                fw_out.get_size(),
                fw_out.get_stride(),
            ),
            subgraphs=[
                subgraph,
                mask_graph,
            ],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )

    inputs_for_flex_attention = (
        [
            query,
            key,
            value,
            fw_lse,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )

    input_gen_fns = {
        4: create_num_blocks_fake_generator(kv_indices),
        5: create_indices_fake,
        6: create_num_blocks_fake_generator(full_kv_indices),
        7: create_indices_fake,
    }

    fw_out = autotune_select_algorithm(
        "flex_attention",
        choices,
        inputs_for_flex_attention,
        fw_out,
        input_gen_fns=input_gen_fns,
    )

    if kernel_options["OUTPUT_LOGSUMEXP"]:
        # see NOTE:[TritonTemplates with multiple outputs]
        fw_out.realize()
        fw_lse.realize()

        # need subgraph inputs and outputs to analyze all symints used in flex attention
        fw_out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
            mask_mod_other_buffers
        )
        fw_out.data.data.subgraph_outs = get_fwd_subgraph_outputs(subgraph, mask_graph)

        return (
            fw_out,
            fw_lse,
        )

    else:
        # need subgraph inputs and outputs to analyze all symints used in flex attention
        fw_out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
            mask_mod_other_buffers
        )
        fw_out.data.data.subgraph_outs = get_fwd_subgraph_outputs(subgraph, mask_graph)

        return fw_out


# ---------------------------- Backward HOP Implementation ----------------------------

# NOTE: Unlike the forward implementations, backward is 2 kernels. This is a
# performance optimization that allows us to apply a much simpler memory
# reduction. Rather than attempting to compute the gradients for Q, K, and V in
# a single kernel invocation, we instead split it into two. The dq kernel
# recomputes the forward attention output and uses it to compute dq. The dk_dv
# kernel transposes the loop structure so compute dk and dv efficiently. This
# two kernel structure with recomputation is common in most flash attention
# implementations.


@dataclass
class JointOutputResult:
    """Result of process_joint_outputs containing gradients and tensors"""

    query_grad: TensorBox
    key_grad: TensorBox
    value_grad: TensorBox
    score_mod_subgraph: Any
    mask_mod_subgraph: Any
    score_mod_tangent_inputs: Any
    mask_mod_tangent_inputs: Any


def process_joint_outputs(
    joint_function_id,
    joint_graph,
    joint_graph_buffer,
    placeholder_inps,
    query,
    key,
    value,
    SPARSE_KV_BLOCK_SIZE,
    SPARSE_Q_BLOCK_SIZE,
    score_mod_tangent_names,
):
    """Process joint graph outputs for backward pass

    Extracts and processes the outputs from the joint graph, returning
    gradients and modified subgraphs for score and mask modifications.
    """
    graph_type = "joint_forward_backward"
    suffix = "_backward"
    from ..ir import Subgraph

    # Extract outputs from joint graph
    query_grad = joint_graph_buffer[-3]
    key_grad = joint_graph_buffer[-2]
    value_grad = joint_graph_buffer[-1]

    def partition_modular_backward_graph_from_joint(
        joint_graph, placeholder_inps, SPARSE_KV_BLOCK_SIZE, SPARSE_Q_BLOCK_SIZE
    ):
        """Extract score_mod and mask_mod subgraphs from joint graph"""
        from ..subgraph_lowering import get_node_and_name_list

        # Get placeholder and output information
        placeholder_name_to_node = {
            inp.get_layout().buffer.get_name(): node
            for inp, node in zip(placeholder_inps, joint_graph.graph.nodes)
            if node.op == "placeholder"
        }
        output_nodes, output_names = get_node_and_name_list(
            joint_graph.graph, return_str_list=True
        )

        assert len(output_names) == 5, f"Expected 5 outputs, got {len(output_names)}"
        score_mod_out_node = output_nodes[0]
        mask_mod_out_node = output_nodes[1]

        # Build subgraphs for score and mask modifications
        def build_modification_backward_subgraph(out_node):
            """Build subgraph containing only nodes needed for given output"""
            import torch.fx

            nodes_to_include = {out_node}
            queue = [out_node]

            while queue:
                node = queue.pop(0)
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg not in nodes_to_include:
                        nodes_to_include.add(arg)
                        queue.append(arg)

            # Create new graph
            new_graph = torch.fx.Graph()
            node_map = {}

            # Add placeholders
            for name, orig_node in placeholder_name_to_node.items():
                if orig_node in nodes_to_include:
                    node_map[orig_node] = new_graph.placeholder(name)

            # Copy nodes
            for node in joint_graph.graph.nodes:
                if node in nodes_to_include and node.op not in (
                    "placeholder",
                    "output",
                ):
                    new_args = tree_map(lambda x: node_map.get(x, x), node.args)
                    new_kwargs = tree_map(lambda x: node_map.get(x, x), node.kwargs)
                    node_map[node] = new_graph.node_copy(node, new_args, new_kwargs)

            # Add output
            new_graph.output(node_map[out_node])
            new_graph_module = torch.fx.GraphModule(root={}, graph=new_graph)

            return new_graph_module

        score_mod_graph = build_modification_backward_subgraph(score_mod_out_node)
        mask_mod_graph = build_modification_backward_subgraph(mask_mod_out_node)

        return score_mod_graph, mask_mod_graph

    score_mod_subgraph, mask_mod_subgraph = partition_modular_backward_graph_from_joint(
        joint_graph, placeholder_inps, SPARSE_KV_BLOCK_SIZE, SPARSE_Q_BLOCK_SIZE
    )

    # Create subgraph objects
    score_mod_subgraph = Subgraph(
        name=f"score_mod_subgraph{suffix}",
        graph_module=score_mod_subgraph,
        graphs=[score_mod_subgraph.graph],
        graph_fn=None,
        root_graph_lowering=V.graph,
    )

    mask_mod_subgraph = Subgraph(
        name=f"mask_mod_subgraph{suffix}",
        graph_module=mask_mod_subgraph,
        graphs=[mask_mod_subgraph.graph],
        graph_fn=None,
        root_graph_lowering=V.graph,
    )

    # Extract tangent inputs
    score_mod_tangent_inputs = []
    mask_mod_tangent_inputs = []

    for i, inp in enumerate(placeholder_inps):
        inp_name = inp.get_layout().buffer.get_name()
        if inp_name in score_mod_tangent_names:
            if i < len(joint_graph_buffer):
                score_mod_tangent_inputs.append(joint_graph_buffer[i])
            else:
                mask_mod_tangent_inputs.append(
                    joint_graph_buffer[i - len(score_mod_tangent_names)]
                )

    return JointOutputResult(
        query_grad=query_grad,
        key_grad=key_grad,
        value_grad=value_grad,
        score_mod_subgraph=score_mod_subgraph,
        mask_mod_subgraph=mask_mod_subgraph,
        score_mod_tangent_inputs=score_mod_tangent_inputs,
        mask_mod_tangent_inputs=mask_mod_tangent_inputs,
    )


def validate_joint_graph(joint_graph: torch.fx.Graph):
    """Validates that joint graph has expected structure for flex attention"""
    placeholders = [n for n in joint_graph.nodes if n.op == "placeholder"]
    expected_names = [
        "primals_1",
        "primals_2",
        "primals_3",
        "primals_4",
        "primals_5",
        "primals_6",
        "primals_7",
        "primals_8",
        "tangents_1",
    ]

    assert len(placeholders) >= 9, (
        f"Expected at least 9 placeholders, got {len(placeholders)}"
    )

    for i, (node, expected) in enumerate(zip(placeholders[:9], expected_names)):
        assert node.name == expected, (
            f"Placeholder {i}: expected {expected}, got {node.name}"
        )

    outputs = [n for n in joint_graph.nodes if n.op == "output"]
    assert len(outputs) == 1, f"Expected 1 output node, got {len(outputs)}"

    output_args = outputs[0].args[0]
    assert len(output_args) >= 5, f"Expected at least 5 outputs, got {len(output_args)}"


@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    (
        query,
        key,
        value,
        out,
        lse,
        grad_out,
        fw_subgraph,
        joint_subgraph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        score_mod_tangent_names,
    ) = args

    assert score_mod_other_buffers is not None
    assert mask_mod_other_buffers is not None

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

    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", query.get_dtype(), [math.inf, math.inf]),
            ("b", torch.int64, []),
            ("h", torch.int64, []),
            ("m", torch.int64, []),
            ("n", torch.int64, []),
            ("out", query.get_dtype(), [math.inf, value.get_size()[-1]]),
            ("lse", torch.float32, [math.inf]),
            ("grad_score", query.get_dtype(), [math.inf, math.inf]),
        ]
    ]

    joint_graph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers) + list(mask_mod_other_buffers),
        joint_subgraph,
    )

    # Validate joint graph structure
    validate_joint_graph(joint_subgraph.graph_module.graph)

    # Process outputs to get gradients and subgraphs
    result = process_joint_outputs(
        joint_function_id=2,
        joint_graph=joint_subgraph,
        joint_graph_buffer=joint_graph_buffer,
        placeholder_inps=placeholder_inps,
        query=query,
        key=key,
        value=value,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        score_mod_tangent_names=score_mod_tangent_names,
    )

    # Continue with backward kernel implementations...
    # (Implementation continues with backward grid functions and kernels)

    raise NotImplementedError(
        "Full backward implementation requires additional backend code"
    )
