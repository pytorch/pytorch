# mypy: allow-untyped-defs
""" Triton Implementation of the flex_attention Kernel for short query length (FlexDecoding)"""
from typing import Any, List, Tuple

import sympy

import torch
from torch._inductor.virtualized import V

from .. import config, ir
from ..ir import FixedLayout, FlexibleLayout
from ..lowering import empty, empty_strided, lowerings
from ..runtime.runtime_utils import is_power_of_2, next_power_of_2
from ..select_algorithm import autotune_select_algorithm, TritonTemplate
from .flex_attention import (
    compute_forward_block_mn,
    compute_forward_inner,
    compute_next_offset_func,
    create_indices_fake,
    create_num_blocks_fake_generator,
    maybe_realize,
)


aten = torch.ops.aten
prims = torch.ops.prims


def flex_decoding_grid(batch_size, kv_heads, gqa_group_size, n_keys, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * kv_heads, SPLIT_KV, 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    groups of SPLIT_KV blocks then combine their output to produce the final result.
    """

    return (batch_size * kv_heads, meta["SPLIT_KV"], 1)


flex_decoding_template = TritonTemplate(
    name="flex_decoding",
    grid=flex_decoding_grid,
    source=r"""
    {{def_kernel("Q", "K", "V", "M", "L", "KV_NUM_BLKS", "KV_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax across local KV split, L local sumexp across local KV split
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # BLOCK_M, QK_HEAD_DIM: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of kv splits
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # TILE_KV: length of each local KV split
    # BLOCK_M: block size that Q is padded along seqlen dim.
    # BLOCK_N: block size of K & V along N dimension.
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # SAFE_M_BOUNDARY: Is Q seqlen a multiple of BLOCK_M? If so, we can skip an extra boundary check for loading query.
    # SAFE_N_BOUNDARY: Is KV seqlen a multiple of BLOCK_N? If so, we can skip an extra boundary check for loading key/value.

    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base.
    #
    # SPARSE_KV_BLOCK_SIZE: sparse mask block size along KV seqlen dim.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    #
    #
    # Output: ACC output accumulated across local KV split.

    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define Q Strides
    stride_qz, stride_qh, stride_qg, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}
    stride_mz, stride_mt, stride_mh, stride_mm = {{stride("M")}}
    stride_lz, stride_lt, stride_lh, stride_lm = {{stride("L")}}


    Z = {{size("Q", 0)}}
    ZKV = {{size("K", 0)}}
    HKV = {{size("Q", 1)}}
    G: tl.constexpr = GQA_SHARED_HEADS
    HQ = HKV * G
    Q_LEN = {{size("Q", 3)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    # Make sure each split is a multiple of BLOCK_N
    TILE_KV_OG = tl.cdiv(KV_LEN, SPLIT_KV)
    TILE_KV = tl.cdiv(TILE_KV_OG, BLOCK_N) * BLOCK_N
    TILE_KV_MULTIPLE: tl.constexpr = (TILE_KV // BLOCK_N)

    off_z = tl.program_id(0) // HKV
    off_zkv = off_z % ZKV
    off_hkv = tl.program_id(0) % HKV
    off_t = tl.program_id(1)

    q_offset = off_z * stride_qz + off_hkv * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_z % SPARSE_Z
    # TODO: support masks not broadcasted along the head dimension.
    tl.device_assert(SPARSE_HQ == 1)
    sparse_idx_h = 0

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    SPARSE_KV_BLOCK_CNT = tl.cdiv(KV_LEN, SPARSE_KV_BLOCK_SIZE)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

    # initialize offsets
    tl.device_assert(BLOCK_M % G == 0)
    BLOCK_M_PER_HQ: tl.constexpr = BLOCK_M // G
    off_g = tl.arange(0, G)                                                 # [G]
    offs_g = tl.ravel(tl.broadcast_to(off_g[:, None], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_hq = offs_g + off_hkv * G
    off_m = tl.arange(0, BLOCK_M_PER_HQ)                                    # [BLOCK_M_PER_HQ]
    offs_m = tl.ravel(tl.broadcast_to(off_m[None, :], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_d = tl.arange(0, QK_HEAD_DIM)
    offs_vd = tl.arange(0, V_HEAD_DIM)

    # Get HZ offsets for KV_NUM_BLKS and KV_IDX
    stride_block_z, stride_block_h, stride_block_row, stride_block_col = {{stride("KV_NUM_BLKS")}}
    sparse_block_hz_offset = sparse_idx_z * stride_block_z + sparse_idx_h * stride_block_h
    stride_kv_z, stride_kv_h, stride_kv_row, stride_kv_col = {{stride("KV_IDX")}}
    sparse_idx_hz_offset = sparse_idx_z * stride_kv_z + sparse_idx_h * stride_kv_h

    # Calculate KV blocks that belong this CTA.
    block_n_start = off_t * TILE_KV_MULTIPLE                        # n_offset inside sparse block
    block_n_end = block_n_start + TILE_KV_MULTIPLE                  # end BLOCK_N

    q_range = stride_qg * off_g[:, None, None] + stride_qm * off_m[None, :, None] + stride_qk * offs_d[None, None, :]

    if SAFE_M_BOUNDARY:
        q = tl.load(Q + q_offset + q_range)
    else:
        mask = off_m[None, :, None] < Q_LEN
        q = tl.load(Q + q_offset + q_range, mask)

    q = tl.reshape(q, [BLOCK_M, QK_HEAD_DIM])


    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply both score_mod and mask_mod

    # find first kv block we are loading and the number of blocks we are loading
    # Offset the kv_indices tensor by the correct batch and head
    kv_indices = KV_IDX + sparse_idx_hz_offset
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_block_hz_offset)
    indices_idx = block_n_start // SPARSE_KV_MULTIPLE
    off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
    off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N
    # first kv block we're loading

    # last valid block according to sparse mask
    block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(QK_HEAD_DIM, KV_LEN),                # (d, N)
        strides=(stride_kk, stride_kn),
        offsets=(0, off_n),
        block_shape=(QK_HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(off_n, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM),
        order=(1, 0)
    )
    offs_n = tl.arange(0, BLOCK_N) + off_n

    acc, l_i, m_i = forward_inner(
        {{gen_argdefs()}},
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        # accumulatd values
        acc, l_i, m_i,
        #offsets
        off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
        #block sparse data
        kv_indices, kv_num_blocks,
        block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )


    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        kv_indices = FULL_KV_IDX + sparse_idx_hz_offset
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_block_hz_offset)
        indices_idx = block_n_start // SPARSE_KV_MULTIPLE
        off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
        off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N

        # last valid block according to sparse mask
        block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(QK_HEAD_DIM, KV_LEN),                # (d, N)
            strides=(stride_kk, stride_kn),
            offsets=(0, off_n),
            block_shape=(QK_HEAD_DIM, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(off_n, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM),
            order=(1, 0)
        )
        offs_n = tl.arange(0, BLOCK_N) + off_n

        acc, l_i, m_i = forward_inner(
            {{gen_argdefs()}},
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            # accumulatd values
            acc, l_i, m_i,
            #offsets
            off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
            #block sparse data
            kv_indices, kv_num_blocks,
            block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )

    m_offset = off_t * stride_mt + off_z * stride_mz
    l_offset = off_t * stride_lt + off_z * stride_lz

    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_mh, stride_mm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_lh, stride_lm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )

    # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    m_i = m_i.reshape(G, BLOCK_M_PER_HQ)
    l_i = l_i.reshape(G, BLOCK_M_PER_HQ)
    if SAFE_M_BOUNDARY:
        tl.store(M_block_ptr, m_i)
        tl.store(L_block_ptr, l_i)
    else:
        tl.store(M_block_ptr, m_i, boundary_check=(1,))
        tl.store(L_block_ptr, l_i, boundary_check=(1,))

    # -- store output
    idx_z = off_z
    idx_t = off_t
    idx_hq = off_hkv*G + off_g[:, None, None]
    idx_m = off_m[None, :, None]
    idx_d = offs_vd[None, None, :]

    mask = (idx_m < Q_LEN)
    acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
    {{store_output(("idx_z", "idx_t", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}
 """
    + compute_forward_inner
    + compute_next_offset_func
    + compute_forward_block_mn,
)


def get_split_k(B: int, H: int, Mk: int, SM: int = 128) -> int:
    """Heuristic for the number of splits from xformer"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = SM // bh  # Each SM should at least get one block.
    split_k = max(split_k, 1)

    return split_k


def _get_decoding_default_config(key) -> Tuple[int, int, int]:
    dtype = key.get_dtype()
    head_dim = key.get_size()[-1]
    sm_version = torch.cuda.get_device_capability()
    default_config = (64, 2, 1)
    if sm_version >= (9, 0):
        if head_dim > 128 and dtype == torch.float32:
            return default_config
        if torch.version.hip is None:
            return (64, 2, 3)
        else:
            return (64, 2, 1)
    return default_config


def create_flex_decoding_kernel(*args, **kwargs):
    (
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        score_mod_subgraph,
        mask_mod_subgraph,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,  # full_kv_num_blocks,
        full_kv_indices,  # full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks,
        _,  # full_q_indices,
        _,  # SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        _,
    ) = block_mask

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    assert V.graph.sizevars.evaluate_expr(
        sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)
    ), f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"

    B = Bq
    kernel_options = dict(kernel_options)

    # TODO: Fix flex decoding non-divisible case!
    if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
        kernel_options.setdefault("IS_DIVISIBLE", False)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", True)

    # Calculate GQA head sharing
    gqa_shared_heads = Hq // Hkv
    if not is_power_of_2(gqa_shared_heads):
        raise ValueError(
            "Number of shared query heads sharing the same KV head must be power of 2. "
        )
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Determine if there are "full" blocks where we only need to apply score_mod, and can skip mask_mod
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        # Create a plackeholder full block list in case it is empty
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

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
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    choices: List[Any] = []
    configs: List[Tuple[int, int, int]] = []
    configs.append(_get_decoding_default_config(key))
    # Note: max_autotune is not supported yet. Causes error in lowering the dynamic shape in reduction ops.
    if config.max_autotune:
        configs += [
            (64, 2, 2),
            (32, 2, 3),
            (128, 2, 3),
        ]

        # Use num_stages=1 on ROCm to avoid shmem limitation
        if torch.version.hip:
            configs = [(c[0], c[1], 1) for c in configs]

    # TODO: fix autotuning.

    kernel_options.setdefault("SM_SCALE", scale)
    kernel_options.setdefault("SPLIT_KV", get_split_k(B, Hkv, seq_len_kv))
    MAX_SPLIT_KV = kernel_options["SPLIT_KV"]

    # create config dependent intermediate buffers
    buf_ACC_shape = [B, MAX_SPLIT_KV, Hq, seq_len_q, v_head_dim]
    buf_ML_shape = buf_ACC_shape[:-1]
    buf_M = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,  # The rowmax is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    buf_L = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,  # The intermediate sumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )

    layout_acc = FixedLayout(
        query.get_device(),
        torch.float32,
        buf_ACC_shape,
        FlexibleLayout.contiguous_strides(buf_ACC_shape),
    )

    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim)

    kernel_options.setdefault(
        "BLOCK_M",
        (
            # m
            # if V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-2], 0))
            # else  # Always use a BLOCK_M > 16 before Triton fix https://github.com/triton-lang/triton/pull/4061 is in pin
            max(
                next_power_of_2(
                    V.graph.sizevars.size_hint(
                        seq_len_q, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
                    )
                    * gqa_shared_heads
                ),
                16,
            )
        ),
    )

    query = ir.ExternKernel.realize_input(query)
    stride_b, stride_hq, stride_seq_len_q, stride_qk_head_dim = query.get_stride()

    # Reshape query for GQA: [B, Hq, Mq, D] -> [B, Hkv, G, Mq, D]
    gqa_query_shape = (B, Hkv, gqa_shared_heads, seq_len_q, qk_head_dim)
    gqa_query_stride = (
        stride_b,
        stride_hq * gqa_shared_heads,
        stride_hq,
        stride_seq_len_q,
        stride_qk_head_dim,
    )
    query = lowerings[aten.as_strided](query, gqa_query_shape, gqa_query_stride)

    V.graph.sizevars.guard_leq(
        seq_len_q * gqa_shared_heads, sympy.Integer(kernel_options["BLOCK_M"])
    )

    kernel_options.setdefault(
        "SAFE_M_BOUNDARY",
        ((seq_len_q * gqa_shared_heads) % kernel_options["BLOCK_M"]) == 0,
    )
    # TODO: This feels sketchy
    kernel_options.setdefault("SAFE_N_BOUNDARY", True)
    # Mark SPARSE_KV_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)

    original_kernel_options = kernel_options.copy()
    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    for BLOCK_N, num_warps, num_stages in configs:
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0:
            continue

        cur_kernel_options = original_kernel_options.copy()
        # Performance tuning
        cur_kernel_options.setdefault("BLOCK_N", BLOCK_N)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        # Work around https://github.com/pytorch/pytorch/issues/129625
        if num_stages == 2:
            continue
        flex_decoding_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                buf_M,
                buf_L,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout_acc,
            subgraphs=[
                score_mod_subgraph,
                mask_mod_subgraph,
            ],
            mutated_inputs=[buf_M, buf_L],
            num_stages=num_stages,
            num_warps=num_warps,
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )

    inputs_for_flex_decoding = (
        [
            query,
            key,
            value,
            buf_M,
            buf_L,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )

    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }

    buf_ACC = autotune_select_algorithm(
        "flex_decoding",
        choices,
        inputs_for_flex_decoding,
        layout_acc,
        input_gen_fns=input_gen_fns,
    )

    # Reduction

    g_M = lowerings[aten.max](buf_M, dim=1, keepdim=True)[0]
    # See [Note] Handle fully masked out rows:
    # g_M Is the global max among split kv blocks.
    masked_rows = lowerings[aten.eq](g_M, -float("inf"))
    adj_M = lowerings[aten.sub](buf_M, g_M)
    adj_M = lowerings[aten.where](masked_rows, 0, adj_M)
    alpha = lowerings[aten.exp2](adj_M)

    buf_L = lowerings[aten.mul](buf_L, alpha)
    g_L = lowerings[aten.sum](buf_L, axis=1)
    masked_rows_squeezed = lowerings[aten.squeeze](masked_rows, dim=1)
    g_L = lowerings[aten.where](masked_rows_squeezed, 1.0, g_L)
    logsumexp = lowerings[aten.log2](g_L)
    logsumexp = lowerings[aten.add](logsumexp, lowerings[aten.squeeze](g_M, dim=1))

    alpha_unseq = lowerings[aten.unsqueeze](alpha, 4)
    buf_ACC = lowerings[aten.mul](buf_ACC, alpha_unseq)
    output = lowerings[aten.sum](buf_ACC, axis=1)
    L_unseq = lowerings[aten.unsqueeze](g_L, 3)
    output = lowerings[aten.div](output, L_unseq)
    output = lowerings[prims.convert_element_type](output, query.get_dtype())

    return (
        output,
        logsumexp,
    )
