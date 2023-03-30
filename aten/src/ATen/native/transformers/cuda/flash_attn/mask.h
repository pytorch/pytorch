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

#include <ATen/cuda/CUDAContext.h>

namespace fmha {


template<typename Cta_tile, bool Is_causal=false>
struct Mask {
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    template<typename BInfo>
    __device__ Mask(const BInfo &binfo, int tidx, const int loop_step_idx_ = 0)
        : actual_seqlen_k(binfo.actual_seqlen_k - loop_step_idx_ * Cta_tile::N)
        , loop_step_idx(loop_step_idx_) {

        const int warp = tidx / Cta_tile::THREADS_PER_WARP;
        const int lane = tidx % Cta_tile::THREADS_PER_WARP;

        static_assert(Cta_tile::WARPS_K == 1, "");

        // find the warp in the Cta tile
        const int warp_n = (warp / Cta_tile::WARPS_M);
        const int warp_m = (warp % Cta_tile::WARPS_M);
        // decompose warp into 8x4 tile
        const int quad = lane / 4;
        const int tid = (lane % 4) * 2;
        row = warp_m * 16 + quad;
        col = warp_n * 16 + tid;
    }

    inline __device__ bool is_valid(const int mi, const int ni, const int ii, const int jj) const {

        // ii and jj iterate over the 2x4 fragment
        // const int current_col = (Is_causal ? loop_step_idx * Cta_tile::N : 0) + ni * Mma_tile::N_PER_MMA_PER_CTA + col + (jj & 2) * 4 + (jj & 1);
        const int current_col = ni * Mma_tile::N_PER_MMA_PER_CTA + col + (jj & 2) * 4 + (jj & 1);
        const int current_row = row_offset + ii * 8;
        const bool col_valid = current_col < actual_seqlen_k;
        // const bool col_valid = (ni * Mma_tile::N_PER_MMA_PER_CTA + col + (jj & 2) * 4 + (jj & 1)) < actual_seqlen_k;
        //&& (row + mi * Mma_tile::M_PER_MMA_PER_CTA + ii * 8) < actual_seqlen_k;
        // bool all_valid = Is_causal ? col_valid && (current_col + loop_step_idx * Cta_tile::N <= current_row) : col_valid;
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 1)) {
        //     printf("current_col=%d, current_row=%d, actual_seqlen_k=%d, col_valid=%d, all_valid=%d\n", current_col, current_row, actual_seqlen_k, col_valid, all_valid);
        // }
        return Is_causal ? col_valid && (current_col + loop_step_idx * Cta_tile::N <= current_row) : col_valid;
        // return row_valid && col_valid;
    }

    //BERT Mask: if upper left is invalid, none are valid
    inline __device__ bool any_valid(const int mi, const int ni) const {
        return is_valid(mi, ni, 0, 0) || is_valid(mi, ni, 1, 0);
    }

    inline __device__ void load(const int it) {
        row_offset = it * Cta_tile::M + row;
    }
    int row_offset;

    int row;
    int col;
    const int loop_step_idx;
    const int actual_seqlen_k;
};

}  // namespace fmha
