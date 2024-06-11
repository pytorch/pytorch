# AOT ID: ['1_inference']
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
from torch._inductor.async_compile import AsyncCompile
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
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
@triton.jit
def triton_poi_fused_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)


@triton.jit
def triton_tem_fused_1(arg_Q, arg_K, arg_V, arg_LSE, arg_M, arg_L, arg_ACC, arg_LOCK_RDCT, out_ptr0):
    BLOCK_N : tl.constexpr = 64
    SPLIT_KV : tl.constexpr = 4
    BLOCK_MMODEL : tl.constexpr = 16
    BLOCK_DMODEL : tl.constexpr = 128
    SCORE_MOD_IS_LINEAR : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    M = arg_M
    L = arg_L
    ACC = arg_ACC
    LOCK_RDCT = arg_LOCK_RDCT

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax, L sumexp, ACC accumulated output
    # reduction lock: LOCK_RDCT. 
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # BLOCK_MMODLE, BLOCK_DMODEL: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of tiles per query
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # BLOCK_N: block size of K & V along N dimension. 
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    stride_qz = 2048
    stride_qh = 128
    stride_qm = 128
    stride_qk = 1
    # Define K Strides
    stride_kz = 65536
    stride_kh = 128
    stride_kn = 128
    stride_kk = 1
    # Define V Strides
    stride_vz = 65536
    stride_vh = 128
    stride_vk = 128
    stride_vn = 1
    # Define M Strides
    stride_mz = 1024
    stride_mh = 1024
    stride_mt = 16
    stride_mm = 1
    # Define L Strides
    stride_lz = 1024
    stride_lh = 1024
    stride_lt = 16
    stride_lm = 1
    # Define ACC Strides
    stride_accz = 131072
    stride_acch = 131072
    stride_acct = 2048
    stride_accm = 128
    stride_accd = 1


    Z = 128
    H = 1
    Q_CTX = 16
    N_CTX = 512
    TILE_KV = N_CTX // SPLIT_KV # lenth of key/value assigned to a single CTA


    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    off_hz = tl.program_id(0)           
    off_t = tl.program_id(1) 
    off_n = off_t * TILE_KV 

    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),        # (M, d)
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),                     # No offset: one CTA per query
        block_shape=(BLOCK_MMODEL, BLOCK_DMODEL),
        order=(1, 0)
    )

    kv_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),                # (d, N)
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
        shape=(SPLIT_KV, Q_CTX),                      # (T, M) 
        strides=(stride_mt, stride_mm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_MMODEL),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + ml_offset,
        shape=(SPLIT_KV, Q_CTX),                      # (T, M)
        strides=(stride_lt, stride_lm),
        offsets=(off_t, 0),
        block_shape=(1, BLOCK_MMODEL),
        order=(1, 0)
    )

    acc_offset = off_hz * stride_acch
    ACC_block_ptr = tl.make_block_ptr(
        base=ACC + acc_offset,
        shape=(SPLIT_KV, Q_CTX, BLOCK_DMODEL),          # (T, M, D)
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
        # qk = q.to(MATMUL_PRECISION)[:, :, None]*k.to(MATMUL_PRECISION)[None, :, :]
        # qk += tl.sum(q.to(MATMUL_PRECISION)[:, :, None]*k.to(MATMUL_PRECISION)[None, :, :], axis=-2)
        qk = tl.dot(q, k.to(MATMUL_PRECISION)).to(tl.float32)

        tl.static_print("qk type", qk.dtype)
 # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        # m = offs_m[:, None]
        # n = start_n + offs_n[None, :]
        # post_mod_scores = (qk)

        # TODO: In the case that score_mod is linear, this can be LICMed
        # if not SCORE_MOD_IS_LINEAR:
        #     post_mod_scores *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        tl.static_print("rowmax type", row_max.dtype)
        tl.static_print("m_i type", m_i.dtype)
        m_i_new = tl.maximum(m_i, row_max)

        # alpha = tl.math.exp2(m_i - m_i_new)
        # p = tl.math.exp2(qk - m_i_new[:, None])
        # if not ROWS_GUARANTEED_SAFE:
        #     masked_out_rows = (m_i_new == float("-inf"))
        #     alpha = tl.where(masked_out_rows, 0, alpha)
        #     p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        # acc *= alpha[:, None]
        # acc += tl.sum(p.to(MATMUL_PRECISION)[:, :, None] * v.to(MATMUL_PRECISION), axis=-2).to(tl.float32)

        # -- update m_i and l_i --
        # l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

#     # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    tl.store(M_block_ptr, m_i[None, :])
    # tl.store(L_block_ptr, l_i[None, :])
    # tl.store(ACC_block_ptr, acc[None, :, :])

    # # Reduce over T for M, L and ACC
    # my_ticket = tl.atomic_add(LOCK_RDCT + off_hz, 1)
    # if my_ticket >= SPLIT_KV - 1: # Last CTA is responsible for the reduction
    #     offs_t = tl.arange(0, SPLIT_KV)
    #     index_ml = offs_t[:, None] * BLOCK_MMODEL + offs_m[None, :] # [T, M]
    #     index_acc = offs_t[:, None, None] *BLOCK_DMODEL*BLOCK_MMODEL + offs_m[None, :, None]*BLOCK_DMODEL + offs_d[None, None, :] #[T, M, D]
    #     t_m_i = tl.load(M+ml_offset+(index_ml))
    #     t_l_i = tl.load(L+ml_offset+(index_ml))
    #     t_acc = tl.load(ACC+acc_offset+(index_acc))

    #     # initialize global matrix. 


    #     # find global rowmax
    #     g_m = tl.max(t_m_i, 0) # [M]

    #     # rebase to new global rowmax
    #     alpha = tl.exp2(t_m_i - g_m[None, :]) # [T, M]
    #     t_l_i = t_l_i * alpha
    #     t_acc *= alpha[:, :, None]  

    #     # reduction for acc and l_i 
    #     g_acc = tl.zeros([BLOCK_MMODEL, BLOCK_DMODEL], dtype=tl.float32)
    #     g_l = tl.zeros([BLOCK_MMODEL], dtype=tl.float32)
    #     g_acc = tl.sum(t_acc, 0)
    #     g_l = tl.sum(t_l_i, 0)
    #     g_acc = g_acc / g_l[:, None]

    #     idx_z = off_hz // H
    #     idx_h = off_hz % H
    #     idx_m = offs_m[:, None]
    #     idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    #     # TODO generalize and add proper mask support
    #     mask = (idx_m != -1) & (idx_d != -1)
    #     xindex = idx_d + (128*idx_m) + (2048*idx_h) + (2048*idx_z)
    #     tl.store(out_ptr0 + (tl.broadcast_to(idx_d + (128*idx_m) + (2048*idx_z), mask.shape)), g_acc, mask) 
    #     # indentation hack https://github.com/pytorch/pytorch/pull/125515


    #     # TODO dont want to write this if we dont require grad
    #     if OUTPUT_LOGSUMEXP:
    #         l_ptrs = LSE + off_hz * Q_CTX + offs_m
    #         lse = g_m + tl.math.log2(g_l)
    #         tl.store(l_ptrs, lse)
import torch._inductor.kernel.flex_decoding
meta0 = {'BLOCK_N': 64, 'SPLIT_KV': 4, 'BLOCK_MMODEL': 16, 'BLOCK_DMODEL': 128, 'SCORE_MOD_IS_LINEAR': False, 'ROWS_GUARANTEED_SAFE': False, 'OUTPUT_LOGSUMEXP': True}


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 1, 16, 128), (2048, 128, 128, 1))
    assert_size_stride(arg1_1, (128, 1, 512, 128), (65536, 128, 128, 1))
    assert_size_stride(arg2_1, (128, 1, 512, 128), (65536, 128, 128, 1))
    buf0 = empty_strided_cuda((128, 1, 16), (16, 16, 1), torch.float32)
    buf1 = empty_strided_cuda((128, 1, 64, 16), (1024, 1024, 16, 1), torch.float32)
    buf2 = empty_strided_cuda((128, 1, 64, 16), (1024, 1024, 16, 1), torch.float32)
    buf3 = empty_strided_cuda((128, 1, 64, 16, 128), (131072, 131072, 2048, 128, 1), torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((128, 1), (1, 1), torch.int32)
        # Source Nodes: [flex_attention], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0[grid(128)](buf4, 128, 128)
        buf5 = empty_strided_cuda((128, 1, 16, 128), (2048, 2048, 128, 1), torch.float16)
        # Source Nodes: [flex_attention], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_tem_fused_1[torch._inductor.kernel.flex_decoding.flex_decoding_grid(128, 1, 16, 128, meta0)](arg0_1, arg1_1, arg2_1, buf0, buf1, buf2, buf3, buf4, buf5)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf1
        del buf2
        del buf3
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 1, 16, 128), (2048, 128, 128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((128, 1, 512, 128), (65536, 128, 128, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((128, 1, 512, 128), (65536, 128, 128, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
