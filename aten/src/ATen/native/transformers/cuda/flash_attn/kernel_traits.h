/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>

#include <ATen/native/transformers/cuda/flash_attn/gemm.h>
#include <ATen/native/transformers/cuda/flash_attn/gmem_tile.h>

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int S, int D, int STEP, int WARPS_M, int WARPS_N, uint32_t FLAGS = 0x08u, typename elem_type_=__half>
struct FMHA_kernel_traits {

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = fmha::Cta_tile_extd<STEP, S, D, WARPS_M, WARPS_N, 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = fmha::Cta_tile_extd<STEP, D, S, WARPS_M, 1, WARPS_N>;

    // Do we use one buffer for K and V.
    static constexpr bool SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x08u) != 0u;
    // Do we keep K in registers.
    static constexpr bool K_IN_REGS = (FLAGS & 0x10u) == 0u;
    // Do we keep V in registers.
    static constexpr bool V_IN_REGS = (FLAGS & 0x100u) == 0u;

    // The global memory tile to load Q.
    using Gmem_tile_q = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>;

    // The shared memory tile to swizzle Q.
    // using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 1>;
    using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_B, S, D>;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_b<Cta_tile_p, fmha::Col>;

    // The global memory tile to load V.
    using Gmem_tile_v = fmha::Gmem_tile_qkv<Cta_tile_o, fmha::BITS_PER_ELEMENT_B, S, D>;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Cta_tile_o>;

    // The global memory tile to store O.
    using Gmem_tile_o = fmha::Gmem_tile_o<Cta_tile_o>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o<Cta_tile_o>;;

    // The global memory tile to load/store S.
    using Gmem_tile_s = fmha::Gmem_tile_mma_s<Cta_tile_p>;

    // The shared memory tile to transpose S.
    using Smem_tile_st = fmha::Smem_tile_mma_transposed<Cta_tile_p>;

    using Gmem_tile_do = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>;

    // // The global memory tile to store the accumulated dK and dV
    // // Hack: we set BYTES_PER_LDGS=32 to emulate the access pattern of dK and dV
    // // where there are 16 bits per lements and 16 bytes per load. In reality we won't
    // // be issue any load or store of size 32 bytes.
    // using Gmem_tile_dkv_accum = fmha::Gmem_tile_qkv<Cta_tile_o, 32, S, D, 32>;

    // The global memory tile to store the softmax sum.
    using Gmem_softmax_sum = fmha::Gmem_summary_stats<Cta_tile_p>;

    // The shared memory tile to store dp sum.
    using Smem_dp_sum = fmha::Smem_tile_dp_sum<Gmem_tile_q, 2>;

    using elem_type = elem_type_;

    // Make sure the number of threads match.
    static_assert((int)Gmem_tile_o::THREADS_PER_ROW == (int)Smem_tile_o::THREADS_PER_ROW, "");

    // The number of threads.
    static constexpr int THREADS = Cta_tile_p::THREADS_PER_CTA;
    // Make sure the number of threads matches both CTAs.
    static_assert(THREADS == Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    static constexpr int BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE;
    // The extra amount of shared memory needed to load V.
    static constexpr int BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE;
    // The amount of shared memory needed for Q, K and V..
    static constexpr int BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V;
    // The amount of shared memory needed to load Q and store O.
    static constexpr int BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE;

    // The amount of shared memory needed for Q, K, V and O.
    static constexpr int BYTES_PER_SMEM = fmha::MaxConstexpr(BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO);
    // Make sure we have enough shared memory.
    static_assert(Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE <= BYTES_PER_SMEM, "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////
