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

#pragma once

#include <ATen/native/transformers/cuda/flash_attn/utils.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/layout/layout.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The number of rows in the CTA tile.
    int M_,
    // The number of cols in the CTA tile.
    int N_,
    // The number of elements in the the K dimension of the GEMM loop.
    int K_,
    // The number of rows of warps.
    int WARPS_M_,
    // The number of cols of warps.
    int WARPS_N_,
    // The number of warps in the K dimension of the GEMM loop.
    int WARPS_K_>
struct Cta_tile_ {

    static constexpr int M = M_, N = N_, K = K_;
    // The number of warps.
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, WARPS_K = WARPS_K_;
    // The number of warps per CTA.
    static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N * WARPS_K;
    // The number of threads per warp.
    static constexpr int THREADS_PER_WARP = 32;
    // The number of threads per CTA.
    static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile>
struct Hmma_tile {
    // The number of elements computed with a single warp-MMA.
    static constexpr int M_PER_MMA = 16, N_PER_MMA = 16, K_PER_MMA = 16;

    // The number of elements computed with a single CTA-MMA.
    static constexpr int M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARPS_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARPS_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARPS_K;

    // The number of MMAs needed to compute the GEMM.
    static constexpr int MMAS_M = DivUpConstexpr(Cta_tile::M, M_PER_MMA_PER_CTA),
        MMAS_N = DivUpConstexpr(Cta_tile::N, N_PER_MMA_PER_CTA),
        MMAS_K = DivUpConstexpr(Cta_tile::K, K_PER_MMA_PER_CTA);

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int M, int N, int K, int WARPS_M, int WARPS_N, int WARPS_K>
using Cta_tile_extd = Cta_tile_<M, N, K, WARPS_M, WARPS_N, WARPS_K>;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
