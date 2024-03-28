""" Triton Implementation of the Templated SDPA Kernel"""
import logging

import torch
from ..select_algorithm import TritonTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


def sdpa_grid(batch_size, num_heads, num_queries, d_model, meta):
    """#How is this kernel parallelized?
    We launch Bsz * num_heads independent programs and
    they work work over blocks of queries
    """
    import triton

    return (triton.cdiv(num_queries, meta["BLOCK_M"]), batch_size * num_heads, 1)


sdpa_template = TritonTemplate(
    name="sdpa",
    grid=sdpa_grid,
    source=r"""
{{def_kernel("Q", "K", "V")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head

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

    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    N_CTX = {{size("Q", 2)}}

    # TODO these need to be plumbed correctly
    IS_CAUSAL = False
    qk_scale = 1.0
    MATMUL_PRECISION = tl.float16

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # TODO fix me
    # qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.to(MATMUL_PRECISION))
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        {{ modification(
            score="qk",
            b="off_hz // H",
            h="off_hz % H",
            m="offs_m[:, None]",
            n="start_n + offs_n[None, :]",
            out="qk"
        ) | indent_except_first(2) }}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        masked_out_rows = (row_max == float("-inf"))
        m_i_new = tl.maximum(m_i, row_max)

        # TODO FIX ME and use 2^x instead of exp
        # alpha = tl.math.exp2(m_i - m_i_new)
        # p = tl.math.exp2(qk - m_i_new[:, None])
        alpha = tl.math.exp(m_i - m_i_new)
        alpha = tl.where(masked_out_rows, 0, alpha)
        p = tl.math.exp(qk - m_i_new[:, None])
        p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION))

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]
    # TODO For backward support we need to add the Logsumexp
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    # O_block_ptr = tl.make_block_ptr(
    #     base=Out + qkv_offset,
    #     shape=(N_CTX, BLOCK_DMODEL),
    #     strides=(stride_om, stride_on),
    #     offsets=(start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    # tl.store(O_block_ptr, acc.to(tl.float16))

    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc")}}
 """,
)
