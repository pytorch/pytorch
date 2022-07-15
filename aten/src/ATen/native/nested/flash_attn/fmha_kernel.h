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

#include <ATen/native/nested/flash_attn/philox.cuh>

#include <ATen/native/nested/flash_attn/fmha_utils.h>
#include <ATen/native/nested/flash_attn/fmha.h>

#include <ATen/native/nested/flash_attn/smem_tile.h>
#include <ATen/native/nested/flash_attn/gmem_tile.h>
#include <ATen/native/nested/flash_attn/mask.h>
#include <ATen/native/nested/flash_attn/softmax.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_CTA>
struct BlockInfoPadded {

    template<typename Params>
    __device__ BlockInfoPadded(const Params &params,
                               const int bidb,
                               const int bidh,
                               const int tidx)
        : bidb(bidb), bidh(bidh), h(params.h) {

        // The block index.
        sum_s_k = params.cu_seqlens_k[bidb];
        actual_seqlen_k = params.cu_seqlens_k[bidb + 1] - sum_s_k;
        sum_s_q = params.cu_seqlens_q[bidb];
        actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - sum_s_q;

        tidx_global = (bidb * params.h + bidh) * THREADS_PER_CTA + tidx;
    }

    __device__ bool stop_early(const int start_col = 0) const {
        return actual_seqlen_k <= start_col;
    }

    int actual_seqlen_q;
    int actual_seqlen_k;
    int sum_s_q;
    int sum_s_k;
    int bidh;
    int bidb;
    int tidx_global;
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int CHUNKS, typename Cta_tile>
struct Noloop_traits{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum{ STEP = Cta_tile::M };
    enum{ SEQLEN = Cta_tile::N };

    template<typename Block_info>
    inline __device__ Noloop_traits(const int bidc, const Block_info& binfo)
        : bidc_(bidc) {
        const int seqlen = binfo.actual_seqlen;
        const int steps = (seqlen  + STEP - 1) / STEP;
        const int steps_per_chunk = (steps + CHUNKS - 1) / CHUNKS;

        const int step_begin = bidc_ * steps_per_chunk;
        const int step_end = min(steps, (bidc_ + 1) * steps_per_chunk);
        const int actual_steps = max(0, step_end - step_begin);
        loop_offset_ = step_begin;
        num_steps_ = actual_steps;

    }

    template<typename ... Tiles>
    inline __device__ void move_all(Tiles & ... tiles) const {
        using expand_type = int[];
        for( int s = 0; s < loop_offset_; s++ ) {
            expand_type{ (tiles.move(), 0)... };
        }
    }

    inline __device__ int get_idx_dk() const {
        //return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const {
        //return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(const int l) {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    const uint32_t bidc_;
    int loop_offset_;
    int num_steps_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
std::tuple<int , int, int, int, int, int> work_dist(const int total_ctas, const int heads_total) {

    constexpr int STEPS_PER_HEAD = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;

    const int num_full_heads = heads_total / total_ctas;
    const int heads_last_wave = heads_total % total_ctas;

    int num_main_groups = 0;
    int main_steps = 0;
    int rest_steps = 0;
    if( heads_last_wave > 0 ) {
        // Number of CTA groups that process within heads.
        num_main_groups = total_ctas / heads_last_wave;
        // Remaining CTAs that process between heads.
        const int rest_ctas = total_ctas - (heads_last_wave * num_main_groups);
        if(rest_ctas == 0) {
            // We have exactly "num_main_groups" CTAs to process each of the remaining heads.
            main_steps = (STEPS_PER_HEAD + num_main_groups - 1) / num_main_groups;
            num_main_groups = STEPS_PER_HEAD / main_steps; // Here: main_step > 0
            rest_steps = STEPS_PER_HEAD % main_steps;

        } else {
            // Ideal number of steps if we could load-balance as evenly as possible.
            const int steps_ideal = (heads_last_wave * STEPS_PER_HEAD + total_ctas - 1) / total_ctas;
            // Iterations that a "rest" CTA has to do at most.
            const int max_rest_iters = (heads_last_wave + rest_ctas - 1) / rest_ctas;
            // Find the first step distribution, s.t. the maximum work of the "rest" CTAs is less than the work of the main CTAs.
            main_steps = steps_ideal;
            rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
            for( ; main_steps * num_main_groups < STEPS_PER_HEAD; main_steps++ ) {
                rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
                const int max_rest_total_steps = rest_steps * max_rest_iters;
                if( max_rest_total_steps < main_steps )
                    break;
            }
            rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
        }
    }

    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    const int max_steps = STEPS_PER_HEAD * num_full_heads + std::max(main_steps, rest_steps);
    const int elts_per_thread_per_step = Mma_tile_p::MMAS_M * Mma_tile_p::MMAS_N * 8;
    const int elts_per_thread = max_steps * elts_per_thread_per_step;

    return {num_full_heads, num_main_groups, heads_last_wave, main_steps, rest_steps, elts_per_thread};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
