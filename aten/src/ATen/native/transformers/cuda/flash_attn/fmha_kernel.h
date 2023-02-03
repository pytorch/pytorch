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

#include <ATen/native/transformers/cuda/flash_attn/fmha.h>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>
#include <ATen/native/transformers/cuda/flash_attn/smem_tile.h>
#include <ATen/native/transformers/cuda/flash_attn/gmem_tile.h>
#include <ATen/native/transformers/cuda/flash_attn/mask.h>
#include <ATen/native/transformers/cuda/flash_attn/softmax.h>
#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>

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

}  // namespace fmha
