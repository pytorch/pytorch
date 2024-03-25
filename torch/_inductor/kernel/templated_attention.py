""" Triton Implementation of the Templated SDPA Kernel"""
import logging

import torch
from ..select_algorithm import TritonTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


def sdpa_grid(S, H, M, D, meta):
    import triton

    return (triton.cdiv(M, meta["BLOCK_M"]), S * H, 1)
    # return (1, 1, 1)


sdpa_template = TritonTemplate(
    name="sdpa",
    grid=sdpa_grid,
    source=r"""
{{def_kernel("Q", "K", "V")}}
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}

    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    N_CTX = {{size("Q", 2)}}

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # Inductor templates don't support writing to block pointers today
    # O_block_ptr = tl.make_block_ptr(
    #     base=out_ptr1 + qvk_offset,
    #     shape=(N_CTX, BLOCK_DMODEL),
    #     strides=(stride_om, stride_on),
    #     offsets=(start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo, hi = 0, N_CTX
    # causal check on every loop iteration can be expensive
    # and peeling the last iteration of the loop does not work well with ptxas
    # so we have a mode to do the causal check in a separate kernel entirely
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # advance block pointers to first iteration of the loop
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        {{ modification(
            score="qk",
            b="off_hz // H",
            h="off_hz % H",
            m="offs_m[:, None]",
            n="start_n + offs_n[None, :]",
            out="qk"
        ) | indent_except_first(2) }}
        # qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        sm_scale = 1
        qk = qk * sm_scale
        m_ij = tl.max(qk, 1)

        is_fully_masked = m_ij == float("-inf")

        p = tl.where(is_fully_masked[:, None], 0, tl.math.exp(qk - m_ij[:, None]))
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_i_new)
        beta = tl.math.exp(m_ij - m_i_new)

        l_i = tl.where(is_fully_masked, l_i, l_i * alpha)
        l_i_new = tl.where(is_fully_masked, l_i, l_i + beta * l_ij)
        # scale p
        p_scale = beta / l_i_new
        p = tl.where(is_fully_masked[:, None], p, p * p_scale[:, None])
        # scale acc
        acc_scale = l_i / l_i_new
        acc = tl.where(is_fully_masked[:, None], acc, acc * acc_scale[:, None])
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)
    # write back O
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc", "mask")}}
 """,
)
