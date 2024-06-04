
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.flex_attention


@triton.jit
def triton_(arg_Q, arg_K, arg_V, arg_LSE, out_ptr0, arg_m, arg_l, arg_acc, arg_rdct):
    BLOCK_N : tl.constexpr = 128
    SPLIT_KV : tl.constexpr = 4
    BLOCK_MMODEL: tl.constexpr = 2
    BLOCK_DMODEL : tl.constexpr = 64
    SCORE_MOD_IS_LINEAR : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    # reduction buffers
    M = arg_m 
    L = arg_l
    ACC = arg_acc
    LOCK_RDCT = arg_rdct

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    # (1024, 128, 64, 1)
    stride_qz = 1024
    stride_qh = 128
    stride_qm = 64
    stride_qk = 1
    # Define K Strides
    # (2097152, 262144, 64, 1)
    stride_kz = 2097152
    stride_kh = 262144
    stride_kn = 64
    stride_kk = 1
    # Define V Strides
    stride_vz = 2097152
    stride_vh = 262144
    stride_vk = 64
    stride_vn = 1
    # Define M Strides
    # (2, 8, 4, 2)
    stride_mz = 64
    stride_mh = 8
    stride_mt = 2
    stride_mm = 1
    # Define L Strides
    # (2, 8, 4, 2)
    stride_lz = 64
    stride_lh = 8
    stride_lt = 2
    stride_lm = 1
    # Define ACC Strides
    # (2, 8, 4, 2, 64)
    stride_accz = 4096
    stride_acch = 512
    stride_acct = 128
    stride_accm = 64
    stride_accd = 1

    Z = 8
    H = 8
    N_CTX = 4096                            # TODO: split among multiple CTAs
    Q_CTX = 2
    TILE_KV = N_CTX // SPLIT_KV
    tl.device_assert(Q_CTX == BLOCK_MMODEL, "Query len must match static kernel parameter BLOCK_MMODEL")

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    off_hz = tl.program_id(0)           
    off_t = tl.program_id(1) 
    off_n = off_t * TILE_KV

    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),        # (M, d) = (2, 64)
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),                     # No offset: one CTA per query
        block_shape=(BLOCK_MMODEL, BLOCK_DMODEL),
        order=(1, 0)
    )

    kv_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),                # (d, N) = (64, 4096)
        strides=(stride_kk, stride_kn),
        offsets=(0, off_t * TILE_KV), 
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(off_t * TILE_KV, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )


    ml_offset = off_hz * stride_mh
    M_block_ptr = tl.make_block_ptr(
        base=M + ml_offset,
        shape=(SPLIT_KV, Q_CTX),                      # (T, M) = (4, 2)
        strides=(stride_mt, stride_mm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_MMODEL),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + ml_offset,
        shape=(SPLIT_KV, Q_CTX),                      # (T, M) = (4, 2)
        strides=(stride_lt, stride_lm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_MMODEL),
        order=(1, 0)
    )

    acc_offset = off_hz * stride_acch
    ACC_block_ptr = tl.make_block_ptr(
        base=ACC + acc_offset,
        shape=(SPLIT_KV, Q_CTX, BLOCK_DMODEL),          # (T, M, D) = (4, 2, 64)
        strides=(stride_acct, stride_accm, stride_accd),
        offsets=(off_t, 0, 0),
        block_shape=(1, BLOCK_MMODEL, BLOCK_DMODEL),
        order=(2, 1, 0)
    )

    # initialize offsets
    offs_m = tl.arange(0, BLOCK_MMODEL)      
    offs_n = tl.arange(0, BLOCK_N)           
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_MMODEL], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_MMODEL], dtype=tl.float32)
    acc = tl.zeros([BLOCK_MMODEL, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = off_n
    hi = lo + TILE_KV
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_MMODEL, BLOCK_N], dtype=tl.float32)
        qk = tl.sum(q[:, :, None]*k[None, :, :], axis=-2)

        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = offs_m[:, None]
        n = start_n + offs_n[None, :]
        tmp0 = (n) - (m)
        tmp1 = tl.abs(tmp0)
        tmp2 = tl.full([1], 2, tl.int32)
        tmp3 = tmp1 == tmp2
        tmp4 = tl.full([1], 1, tl.int32)
        tmp5 = tmp1 == tmp4
        tmp6 = 0.5
        tmp7 = (qk) * tmp6
        tmp8 = tl.where(tmp5, tmp7, (qk))
        tmp9 = 2.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.where(tmp3, tmp10, tmp8)
        qk = tmp11

        # TODO: In the case that score_mod is linear, this can be LICMed
        if not SCORE_MOD_IS_LINEAR:
            qk *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_i_new == float("-inf"))
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc *= alpha[:, None]
        p_ = p.to(MATMUL_PRECISION)[:, :, None] # dependent on this triton fix: https://github.com/htyu/triton/commit/c36c24c3cd5e872cb113f1cc56a46fb962ac4e27
        delta_acc = tl.sum(p_ * v.to(MATMUL_PRECISION), axis=-2)
        acc += delta_acc 

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    tl.store(M_block_ptr, m_i[None, :])
    tl.store(L_block_ptr, l_i[None, :])
    tl.store(ACC_block_ptr, acc[None, :, :])
    
    # Reduction over M, L, & ACC
    my_ticket = tl.atomic_add(LOCK_RDCT + off_hz, 1)
    if my_ticket >= SPLIT_KV - 1:
        offs_t = tl.arange(0, SPLIT_KV)
        index_ml = offs_t[:, None] * BLOCK_MMODEL + offs_m[None, :] # [T, M]
        index_acc = offs_t[:, None, None] *BLOCK_DMODEL*BLOCK_MMODEL + offs_m[None, :, None]*BLOCK_DMODEL + offs_d[None, None, :] #[T, M, D]
        t_m_i = tl.load(M+ml_offset+(index_ml))
        t_l_i = tl.load(L+ml_offset+(index_ml))
        t_acc = tl.load(ACC+acc_offset+(index_acc))

        # find global rowmax
        g_m = tl.max(t_m_i, 0) # [M]

        # rebase to new global rowmax
        alpha = tl.exp2(t_m_i - g_m[None, :]) # [T, M]
        t_l_i = t_l_i * alpha
        t_acc *= alpha[:, :, None]  

        # reduction for acc and l_i 
        g_acc = tl.sum(t_acc, 0)
        g_l = tl.sum(t_l_i, 0)
        g_acc = g_acc / g_l[:, None]

        idx_z = off_hz // H
        idx_h = off_hz % H
        idx_m = offs_m[:, None]
        idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
        xindex = idx_d + (stride_qm*idx_m) + (stride_qh*idx_h) + (stride_qz*idx_z)
        tl.store(out_ptr0 + (xindex), g_acc, None)

        # TODO dont want to write this if we dont require grad
        if OUTPUT_LOGSUMEXP:
            l_ptrs = LSE + off_hz * Q_CTX + offs_m
            lse = g_m + tl.math.log2(g_l)
            tl.store(l_ptrs, lse)
        


meta0 = {'BLOCK_N': 128, 'SPLIT_KV': 4, 'BLOCK_DMODEL': 64, 'BLOCK_MMODEL':2, 'SCORE_MOD_IS_LINEAR': False, 'ROWS_GUARANTEED_SAFE': False, 'OUTPUT_LOGSUMEXP': True}


async_compile.wait(globals())
del async_compile


def sdpa_decoding_grid(batch_size, num_heads, n_keys, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_keys, key_tile_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    """
    import triton

    return (batch_size * num_heads, meta['SPLIT_KV'], 1)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    buf0 = torch.zeros(2, 8, 2, device='cuda', dtype=torch.float32) #LSE [B, H, M]
    buf_m = torch.zeros(2, 8, 4, 2, device='cuda', dtype=torch.float32) # rowmax reduction buffer [B, H, SPLIT_KV, M]
    buf_l = torch.zeros(2, 8, 4, 2, device='cuda', dtype=torch.float32) # l_i reduction buffer [B, H, SPLIT_KV, M]
    buf_acc = torch.zeros(2, 8, 4, 2, 64, device='cuda', dtype=torch.float32) # acc reduction buffer [B, H, SPLIT_KV, M, D]
    lock_rdct = torch.zeros(2, 8, device='cuda', dtype=torch.int32) # lock for reduction [B, H]

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = torch.full((2, 8, 2, 64), float('nan'), device='cuda', dtype=torch.float32) # output
        # Source Nodes: [flex_attention], Original ATen: []
        grid=sdpa_decoding_grid(2, 8, 4096, 64, meta0)
        print(grid)
        triton_[grid](arg0_1, arg1_1, arg2_1, buf0, buf1, buf_m, buf_l, buf_acc, lock_rdct)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf_m
        del buf_l
        del buf_acc
        del lock_rdct
    return (buf1, )


from torch.nn.attention._flex_attention import _flex_attention as flex_attention
from flex_decoder import eager_flash_decoder


def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) == 2, score * 2.0, score)
    return score

def run_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    arg0_1 = torch.rand((2, 8, 2, 64), device='cuda', dtype=torch.float32) # Q
    arg1_1 = torch.rand((2, 8, 4096, 64), device='cuda:0', dtype=torch.float32) # K
    arg2_1 = torch.rand((2, 8, 4096, 64), device='cuda:0', dtype=torch.float32) # V
    fn = lambda: call([arg0_1, arg1_1, arg2_1])


    # gold_results, _ = eager_flash_decoder(arg0_1, arg1_1, arg2_1, checkerboard, Bc=1024)
    gold_results = flex_attention(arg0_1, arg1_1, arg2_1, checkerboard)
    print("gold output", gold_results)
    (triton_results,) = fn()
    print("triton output", triton_results)
    torch.testing.assert_close(triton_results, gold_results, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_module()
