
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


# kernel path: /tmp/torchinductor_joydong/fw/cfw7br3v2su43oqa2vc5oyacpep6phlronti75r4565sx7qwvq2i.py
# Source Nodes: [flex_attention], Original ATen: []
# flex_attention => flex_attention
triton_tem_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_0', 'backend_hash': '3d930ae76fa5e25713342ca2a1b35b7ee99023f455bffb679bcc13dbe5fe0c6e', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_Q, arg_K, arg_V, arg_LSE, out_ptr0):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_DMODEL : tl.constexpr = 64
    SCORE_MOD_IS_LINEAR : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_M
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    stride_qz = 4096
    stride_qh = 512
    stride_qm = 64
    stride_qk = 1
    # Define K Strides
    stride_kz = 2097152
    stride_kh = 262144
    stride_kn = 64
    stride_kk = 1
    # Define V Strides
    stride_vz = 2097152
    stride_vh = 262144
    stride_vk = 64
    stride_vn = 1

    Z = 2
    H = 8
    N_CTX = 8

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

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

    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.to(MATMUL_PRECISION), acc=qk)
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = offs_m[:, None]
        n = start_n + offs_n[None, :]
        tmp0 = (n) - (m)
        tmp1 = tl_math.abs(tmp0)
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
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION), acc)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store output and logsumexp
    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]

    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    xindex = idx_d + (64*idx_m) + (512*idx_h) + (4096*idx_z)
    tl.store(out_ptr0 + (xindex), acc, None)

    # TODO dont want to write this if we dont require grad
    if OUTPUT_LOGSUMEXP:
        l_ptrs = LSE + off_hz * N_CTX + offs_m
        lse = m_i + tl.math.log2(l_i)
        tl.store(l_ptrs, lse)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.flex_attention
meta0 = {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_DMODEL': 64, 'SCORE_MOD_IS_LINEAR': False, 'ROWS_GUARANTEED_SAFE': False, 'OUTPUT_LOGSUMEXP': True}


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 8, 8, 64), (4096, 512, 64, 1))
    assert_size_stride(arg1_1, (2, 8, 4096, 64), (2097152, 262144, 64, 1))
    assert_size_stride(arg2_1, (2, 8, 4096, 64), (2097152, 262144, 64, 1))
    buf0 = empty_strided_cuda((2, 8, 8), (64, 8, 1), torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((2, 8, 8, 64), (4096, 512, 64, 1), torch.float32)
        # Source Nodes: [flex_attention], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_tem_fused_0.run(arg0_1, arg1_1, arg2_1, buf0, buf1, grid=torch._inductor.kernel.flex_attention.sdpa_grid(2, 8, 8, 64, meta0), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 8, 8, 64), (4096, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 8, 4096, 64), (2097152, 262144, 64, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 8, 4096, 64), (2097152, 262144, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
