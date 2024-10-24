"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

import triton
import triton.language as tl

import itertools

def gen_configs():
    products = itertools.product([32, 64, 128],
                                 [32, 64, 128],
                                 [32, 64, 128],
                                 # [54, 84, 108, 128, 216, 432, 864],
                                 [1, 2, 4, 8, 16])
    configs=[]
    # for BLOCK_SIZE_K, NUM_SM, num_warps in products:
    for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps in products:
        configs += [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': BLOCK_SIZE_K,
            # 'NUM_SM': NUM_SM,
            'num_stages': num_warps,
            'GROUP_SIZE_M': 8,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': BLOCK_SIZE_K,
            # 'NUM_SM': NUM_SM,
            'num_stages': num_warps,
            'GROUP_SIZE_M': 8,
        }),
        ]
    return configs

@triton.autotune(
    configs=gen_configs(),
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    b_ptr,
    group_size,
    gn,
    gk,
    ldb,
    a_offsets_ptr,
    a_ptr,
    lda,
    ldc,
    c_ptr,
    max_M,
    # # number of virtual SM
    # NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(gn, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tile_m_idx = pid_m
    tile_n_idx = pid_n

    # tile_m_idx = tl.program_id(0)
    # tile_n_idx = tl.program_id(1)

    batch_id = tl.program_id(1)

    # get the gemm size of the current problem
    a_offset_0 = tl.load(a_offsets_ptr + batch_id, eviction_policy='evict_last')
    a_offset_1 = tl.load(a_offsets_ptr + batch_id + 1, eviction_policy='evict_last')
    gm = a_offset_1 - a_offset_0
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    if tile_m_idx < num_m_tiles:
        # pick up a tile from the current gemm problem
        # figure out tile coordinates

        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        offs_am = offs_am % gm
        offs_bn = offs_bn % gn

        # Rematerialize on each loop
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + a_offset_0 * lda + (offs_am[:, None]) * lda + offs_k[None, :]
        b_ptrs = b_ptr + batch_id * gk * gn + offs_k[:, None] * ldb + (offs_bn[None, :])
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
            # # # hint to Triton compiler to do proper loop pipelining
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])
            a = tl.load(a_ptrs, mask=offs_k[None, :] < gk - kk * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < gk - kk * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b) #, accumulator) #, "ieee")
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * ldb
        c = accumulator.to(tl.float16)

        # offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Trying to save registers by recomputing indices
        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        a_offset_0 = tl.load(a_offsets_ptr + batch_id, eviction_policy='evict_last')
        a_offset_1 = tl.load(a_offsets_ptr + batch_id + 1, eviction_policy='evict_last')
        gm = a_offset_1 - a_offset_0

        c_ptrs = c_ptr + a_offset_0 * ldc + ldc * offs_am[:, None] + offs_bn[None, :]
        c_mask = (offs_am[:, None] < gm) & (offs_bn[None, :] < gn)

        # assumes full tile for now
        tl.store(c_ptrs, c, mask=c_mask)

        # # go to the next tile by advancing NUM_SM
        # tile_idx += NUM_SM

def group_gemm_fn(tensor_a, tensor_b):
    assert tensor_a.is_nested
    assert not tensor_b.is_nested
    assert tensor_a.size(0) == tensor_b.size(0)
    group_size = tensor_a.size(0)

    assert tensor_b.is_contiguous()

    a_values = tensor_a.values()
    a_offsets = tensor_a.offsets().to(torch.int32)

    assert a_values.dim() == 2

    B, K, N = tensor_b.shape

    c_values = a_values.new_empty((a_values.size(0), N))
    c_offsets = a_offsets

    max_M = tensor_a._max_seqlen
    # we use a fixed number of CTA, and it's auto-tunable
    # grid = lambda META: (META['NUM_SM'], group_size)
    grid = lambda META: (triton.cdiv(max_M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                         group_size)
    grouped_matmul_kernel[grid](
        tensor_b,
        group_size,
        N,
        K,
        tensor_b.stride(1),
        a_offsets,
        a_values,
        a_values.stride(0),
        c_values.stride(0),
        c_values,
        max_M,
        # BLOCK_SIZE_M=128,
        # BLOCK_SIZE_N=128,
        # BLOCK_SIZE_K=128,
        # NUM_SM=128,
        # num_stages=1,
    )

    return c_values
