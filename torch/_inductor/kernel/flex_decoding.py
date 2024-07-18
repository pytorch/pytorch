# mypy: allow-untyped-defs
""" Triton Implementation of the flex_attention Kernel for short query length (FlexDecoding)"""
from typing import Any, List, Tuple

import sympy

import torch
from torch._inductor.virtualized import V
from ..ir import FixedLayout, FlexibleLayout
from ..lowering import empty_strided, lowerings
from ..runtime.runtime_utils import next_power_of_2
from ..select_algorithm import autotune_select_algorithm, TritonTemplate


aten = torch.ops.aten
prims = torch.ops.prims


def flex_decoding_grid(batch_size, num_heads, n_keys, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, SPLIT_KV, 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    groups of SPLIT_KV blocks then combine their output to produce the final result.
    """

    return (batch_size * num_heads, meta["SPLIT_KV"], 1)


flex_decoding_template = TritonTemplate(
    name="flex_decoding",
    grid=flex_decoding_grid,
    source=r"""
    {{def_kernel("Q", "K", "V", "M", "L", "KV_NUM_BLKS", "KV_IDX")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax across local KV split, L local sumexp across local KV split
    # M: Number of queries, N: Number of keys/values, D(BLOCK_DMODEL): Model dimension
    # BLOCK_M, BLOCK_DMODEL: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of kv splits
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # TILE_KV: length of each local KV split
    # BLOCK_M: block size that Q is padded along seqlen dim.
    # BLOCK_N: block size of K & V along N dimension.
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


    # Define Q Strides
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    # Define K Strides
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    # Define V Strides
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}
    # Define M Strides
    stride_mz = {{stride("M", 0)}}
    stride_mh = {{stride("M", 1)}}
    stride_mt = {{stride("M", 2)}}
    stride_mm = {{stride("M", 3)}}
    # Define L Strides
    stride_lz = {{stride("L", 0)}}
    stride_lh = {{stride("L", 1)}}
    stride_lt = {{stride("L", 2)}}
    stride_lm = {{stride("L", 3)}}


    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    KV_LEN = {{size("K", 2)}}
    # Make sure each split is a multiple of BLOCK_N
    TILE_KV_OG = tl.cdiv(KV_LEN, SPLIT_KV)
    TILE_KV = tl.cdiv(TILE_KV_OG, BLOCK_N) * BLOCK_N

    TILE_KV_MULTIPLE: tl.constexpr = (TILE_KV // BLOCK_N)

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_H = {{size("KV_NUM_BLKS", 1)}}

    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    SPARSE_KV_BLOCK_CNT: tl.constexpr = KV_LEN // SPARSE_KV_BLOCK_SIZE

    MATMUL_PRECISION = Q.dtype.element_ty

    off_z = tl.program_id(0) // H
    off_h = tl.program_id(0) % H
    off_t = tl.program_id(1)

    sparse_idx_z = off_z % SPARSE_Z
    sparse_idx_h = off_h % SPARSE_H

    sparse_hz_offset = sparse_idx_z * SPARSE_H + sparse_idx_h
    kv_indices = KV_IDX + sparse_hz_offset * SPARSE_KV_BLOCK_CNT
    sparse_block_num = tl.load(KV_NUM_BLKS + sparse_hz_offset)


    block_n_start = off_t * TILE_KV_MULTIPLE                       # n_offset inside sparse block
    block_n_end = block_n_start + TILE_KV_MULTIPLE                 # end BLOCK_N
    block_n_last_valid = sparse_block_num * SPARSE_KV_MULTIPLE     # last valid block according to sparse mask
    block_n_end = block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid


    indices_idx = block_n_start // SPARSE_KV_MULTIPLE
    off_n_block_in_spase = block_n_start % SPARSE_KV_MULTIPLE
    off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_spase * BLOCK_N
    # first kv block we're loading

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_LEN, BLOCK_DMODEL),        # (M, d)
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),                     # No offset: one CTA per query
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, KV_LEN),                # (d, N)
        strides=(stride_kk, stride_kn),
        offsets=(0, off_n),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(KV_LEN, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(off_n, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_offset = off_h * stride_mh + off_z * stride_mz
    l_offset = off_h * stride_lh + off_z * stride_lz
    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(SPLIT_KV, Q_LEN),                      # (T, M)
        strides=(stride_mt, stride_mm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_M),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(SPLIT_KV, Q_LEN),                      # (T, M)
        strides=(stride_lt, stride_lm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_M),
        order=(1, 0)
    )


    # initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + off_n
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if SAFE_M_BOUNDARY:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, ))
    RCP_LN2 = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)


    # loop over k, v and update accumulator
    for start_n in range(block_n_start, block_n_end):
        # -- load k, v --
        k = tl.load(K_block_ptr).to(MATMUL_PRECISION)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        if not PRESCALE_QK:
            qk *= SM_SCALE

        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = offs_m[:, None]
        n = offs_n[None, :]
        # TODO: Add load mask in modification when M/N Boundary is not safe
        {{ modification(
            subgraph_number=0,
            output_name="post_mod_scores",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
            out="qk"
        ) | indent_except_first(2) }}
        # TODO: In the case that score_mod is linear, this can be LICMed
        if not PRESCALE_QK:
            post_mod_scores *= RCP_LN2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(post_mod_scores, 1)
        m_i_new = tl.maximum(m_i, row_max)

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(post_mod_scores - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_i_new == float("-inf"))
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION), acc=acc)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new


        # update pointers
        indices_idx = start_n // SPARSE_KV_MULTIPLE

        cur_block = tl.load(kv_indices + indices_idx, eviction_policy="evict_last")
        next_block = tl.load(kv_indices + indices_idx + 1, eviction_policy="evict_last", mask=indices_idx + 1 < sparse_block_num)
        needs_jump = (start_n + 1) % SPARSE_KV_MULTIPLE == 0
        jump_to_block = (next_block - cur_block ) * SPARSE_KV_BLOCK_SIZE - (SPARSE_KV_MULTIPLE - 1) * BLOCK_N

        offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK_N


        K_block_ptr = tl.advance(K_block_ptr, (0, offset))
        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        offs_n = offs_n + offset

    # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    if SAFE_M_BOUNDARY:
        tl.store(M_block_ptr, m_i[None, :])
        tl.store(L_block_ptr, l_i[None, :])
    else:
        tl.store(M_block_ptr, m_i[None, :], boundary_check=(1,))
        tl.store(L_block_ptr, l_i[None, :], boundary_check=(1,))

    # -- store output
    idx_z = off_z
    idx_h = off_h
    idx_t = off_t
    idx_m = offs_m[:, None]
    idx_d = offs_d[None, :]
    # TODO generalize and add proper mask support
    mask = (idx_m < Q_LEN)
    {{store_output(("idx_z", "idx_h", "idx_t", "idx_m", "idx_d"), "acc", "mask")}}
 """,
)


MAX_SPLIT_KV = 64


def get_split_k(B: int, H: int, Mk: int, SM: int = 128) -> int:
    """Heuristic for the number of splits from xformer"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = SM // bh  # Each SM should at least get one block.
    split_k = max(split_k, 1)

    return split_k


def _get_decoding_default_config(key) -> Tuple[int, int, int]:
    default_config = (64, 2, 3)

    return default_config


def create_flex_decoding_kernel(*args, **kwargs):
    (
        subgraph_buffer,
        query,
        key,
        value,
        subgraph,
        block_mask,
        scale,
        *other_buffers,
    ) = args
    (
        sparse_kv_num_blocks,
        sparse_kv_indices,
        full_kv_num_blocks,  # full_kv_num_blocks,
        _,  # full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks,
        _,  # full_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        _,  # SPARSE_Q_BLOCK_SIZE,
        mask_graph,
    ) = block_mask
    if full_kv_num_blocks is not None:
        raise NotImplementedError("NYI: Flex decoding only supports full mask. ")
    for buf in [
        query,
        key,
        value,
        sparse_kv_num_blocks,
        sparse_kv_indices,
    ]:
        buf.realize()
    choices: List[Any] = []
    configs: List[Tuple[int, int, int]] = []
    configs.append(_get_decoding_default_config(key))
    # Note: max_autotune is not supported yet. Causes error in lowering the dynamic shape in reduction ops.
    # if config.max_autotune:
    #     configs += [
    #         (64, 2, 2),
    #         (32, 2, 3),
    #     ]
    # TODO: fix autotuning.

    SPLIT_KV = get_split_k(key.get_size()[0], key.get_size()[1], key.get_size()[2])
    MAX_SPLIT_KV = SPLIT_KV
    assert SPLIT_KV <= MAX_SPLIT_KV

    # create config dependent intermediate buffers
    buf_ML_shape = query.get_size()[:-2] + [
        MAX_SPLIT_KV,
        query.get_size()[-2],
    ]  # [B, H, SPLIT_KV, M]
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

    buf_ACC_shape = (
        query.get_size()[:-2] + [MAX_SPLIT_KV] + query.get_size()[-2:]
    )  # [B, H, SPLIT_KV, M, D]

    layout_acc = FixedLayout(
        query.get_device(),
        torch.float32,
        buf_ACC_shape,
        FlexibleLayout.contiguous_strides(buf_ACC_shape),
    )

    m = query.get_size()[-2]
    BLOCK_M = (
        # m
        # if V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-2], 0))
        # else  # Always use a BLOCK_M > 16 before Triton fix https://github.com/triton-lang/triton/pull/4061 is in pin
        max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    m, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
                )
            ),
            16,
        )
    )

    V.graph.sizevars.guard_leq(m, sympy.Integer(BLOCK_M))

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    for BLOCK_N, num_warps, num_stages in configs:
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0:
            continue
        flex_decoding_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                buf_M,
                buf_L,
                sparse_kv_num_blocks,
                sparse_kv_indices,
            ],
            layout=layout_acc,
            subgraphs=[
                subgraph_buffer,
            ],
            mutated_inputs=[buf_M, buf_L],
            num_stages=num_stages,
            num_warps=num_warps,
            call_sizes=query.get_size(),
            BLOCK_M=BLOCK_M,
            SPLIT_KV=SPLIT_KV,
            BLOCK_DMODEL=query.get_size()[-1],
            SM_SCALE=scale,
            # Performance tuning
            BLOCK_N=BLOCK_N,
            # For now, we always assume the "sound" option
            ROWS_GUARANTEED_SAFE=False,
            SAFE_M_BOUNDARY=(query.get_size()[-2] % BLOCK_M) == 0,
            SAFE_N_BOUNDARY=True,
            PRESCALE_QK=False,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        )

    inputs_for_flex_decoding = [
        query,
        key,
        value,
        buf_M,
        buf_L,
        sparse_kv_num_blocks,
        sparse_kv_indices,
    ] + list(other_buffers)
    buf_ACC = autotune_select_algorithm(
        "flex_decoding", choices, inputs_for_flex_decoding, layout_acc
    )

    # Reduction

    g_M = lowerings[aten.max](buf_M, dim=-2, keepdim=True)[0]
    adj_M = lowerings[aten.sub](buf_M, g_M)
    alpha = lowerings[aten.exp2](adj_M)

    buf_L = lowerings[aten.mul](buf_L, alpha)
    g_L = lowerings[aten.sum](buf_L, axis=-2)
    logsumexp = lowerings[aten.log2](g_L)
    logsumexp = lowerings[aten.add](logsumexp, lowerings[aten.squeeze](g_M, dim=-2))

    alpha_unseq = lowerings[aten.unsqueeze](alpha, 4)
    buf_ACC = lowerings[aten.mul](buf_ACC, alpha_unseq)
    output = lowerings[aten.sum](buf_ACC, axis=-3)
    L_unseq = lowerings[aten.unsqueeze](g_L, 3)
    output = lowerings[aten.div](output, L_unseq)
    output = lowerings[prims.convert_element_type](output, query.get_dtype())

    return (
        output,
        logsumexp,
    )
