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

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <ATen/native/transformers/cuda/flash_attn/fmha_utils.h>

namespace pytorch_fmha {

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    // size_t qkv_stride_in_elts;
    // size_t qkv_stride_in_bytes;
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t q_row_stride_in_elts;
    uint32_t k_row_stride_in_elts;
    uint32_t v_row_stride_in_elts;
    uint32_t q_head_stride_in_elts;
    uint32_t k_head_stride_in_elts;
    uint32_t v_head_stride_in_elts;

    // The number of heads.
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_fprop_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    // size_t o_stride_in_elts;
    // size_t o_stride_in_bytes;
    uint32_t o_row_stride_in_elts;
    uint32_t o_head_stride_in_elts;
    uint32_t o_tmp_row_stride_in_elts;
    uint32_t o_tmp_head_stride_in_elts;

    // The pointer to the O_tmp matrix, which holds O intermediate value during
    // the loop;
    void *__restrict__ o_tmp_ptr;

    // The pointer to the S matrix.
    void * __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    // int64_t s_stride_in_bytes;
    uint32_t s_stride_in_bytes;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_bmm1f;
    uint32_t scale_bmm1;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int *__restrict__ blockmask;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    uint32_t p_dropout_in_uint;
    uint16_t p_dropout_in_uint16_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_bmm1_rp_dropout;

    // Scale factor of 1 / (1 - p_dropout), in half2.
    uint32_t scale_dropout;

    // Random state.
    at::PhiloxCudaState philox_args;
    int64_t * extragraph_offset;
    int64_t * seed;

    bool is_bf16;
    bool is_causal;

    int num_splits; // How many SMs per attention matrix.
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_dgrad_params : public FMHA_fprop_params {

    // The dQKV matrices.
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q dimension
    // void *__restrict__ dk_accum_ptr;
    // void *__restrict__ dv_accum_ptr;

    // The stride between rows of the dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t dq_row_stride_in_elts;
    uint32_t dk_row_stride_in_elts;
    uint32_t dv_row_stride_in_elts;
    uint32_t dq_head_stride_in_elts;
    uint32_t dk_head_stride_in_elts;
    uint32_t dv_head_stride_in_elts;

    // The dO matrix. We assume it is contiguous.
    void * __restrict__ do_ptr;

    // The pointer to the softmax d sum.
    void * __restrict__ dsoftmax_sum;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_params>
struct Launch_params{
    Launch_params(cudaDeviceProp * props_,
                  cudaStream_t stream_,
                  bool is_dropout_,
                  bool return_softmax_)
        : elts_per_thread(0)
        , props(props_)
        , stream(stream_)
        , is_dropout(is_dropout_)
        , return_softmax(return_softmax_) {
    }

    size_t elts_per_thread;

    cudaDeviceProp * props;

    cudaStream_t stream;

    bool is_dropout;
    bool return_softmax;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_fwd_hdim32(Launch_params<FMHA_fprop_params> &launch_params);
void run_fmha_fwd_hdim64(Launch_params<FMHA_fprop_params> &launch_params);
void run_fmha_fwd_hdim128(Launch_params<FMHA_fprop_params> &launch_params);

void run_fmha_bwd_hdim32(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure);
void run_fmha_bwd_hdim64(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure);
void run_fmha_bwd_hdim128(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure);

void run_fmha_block_fp16_sm80(Launch_params<FMHA_fprop_params> &launch_params, const bool configure);

void run_fmha_block_dgrad_fp16_sm80(const FMHA_dgrad_params &params, cudaStream_t stream);

}; // namespace pytorch_fmha
