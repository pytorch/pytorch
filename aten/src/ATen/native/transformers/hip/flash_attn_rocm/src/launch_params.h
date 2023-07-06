// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <vector>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/hip/HIPGuard.h>
#include <c10/core/DeviceType.h>

#include <ATen/native/transformers/hip/flash_attn_rocm/src/utils.h>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

struct QkvParams {
  // The QKV matrices.
  std::vector<const void*> q_ptr; //changed to ck input type
  std::vector<const void*> k_ptr;
  std::vector<const void*> v_ptr;

  std::vector<at::Tensor> q_tensors;
  std::vector<at::Tensor> k_tensors;
  std::vector<at::Tensor> v_tensors;

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

struct FlashFwdParams : public QkvParams {
  // The O matrix (output).
  // void * __restrict__ o_ptr;
  std::vector<void*> o_ptr;
  
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
  // void * __restrict__ s_ptr;
  std::vector<void*> s_ptr;
  // The stride between rows of the S matrix.
  // int64_t s_stride_in_bytes;
  uint32_t s_stride_in_bytes;

  // The pointer to the softmax sum.
  // void * __restrict__ softmax_lse_ptr;
  std::vector<void*> softmax_lse_ptr;

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

  std::vector<int> host_seqlens_q;
  std::vector<int> host_seqlens_k;

  int num_splits; // How many SMs per attention matrix.

  bool is_bf16;
  bool is_causal;
  bool is_performance_mode;
  bool is_deterministic;
  bool is_using_qloop;
};

struct FlashBwdParams : public FlashFwdParams {
  // The O matrix (output).
  std::vector<const void*> y_ptr;
  std::vector<void*> z_ptr;
  std::vector<const void*> lse_ptr;
  std::vector<const void*> ygrad_ptr;
  std::vector<void*> qgrad_ptr;
  std::vector<void*> kgrad_ptr;
  std::vector<void*> vgrad_ptr;

  std::vector<at::Tensor> qgrad_tensors;
  std::vector<at::Tensor> kgrad_tensors;
  std::vector<at::Tensor> vgrad_tensors;

  // at::Tensor dq_tmp;
  // at::Tensor dk_tmp;
  // at::Tensor dv_tmp;
  // The dimensions.
  int b, seqlen_q, seqlen_k, d;

  // The scaling factors for the kernel.
  float scale_bmm1f;

  // array of length b+1 holding starting offset of each sequence.
  int * __restrict__ cu_seqlens_q;
  int * __restrict__ cu_seqlens_k;

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

  std::vector<int> host_seqlens_q;
  std::vector<int> host_seqlens_k;

  int num_splits; // How many SMs per attention matrix.
};


template<typename KernelParams>
struct LaunchParams{
  LaunchParams(hipDeviceProp_t *props,
               hipStream_t stream,
               bool is_dropout,
               bool return_softmax)
      : elts_per_thread_(0),
        props_(props), 
        stream_(stream), 
        is_dropout_(is_dropout), 
        return_softmax_(return_softmax) {}

  size_t elts_per_thread_;
  hipDeviceProp_t * props_;
  hipStream_t stream_;
  
  bool is_dropout_;
  bool return_softmax_;

  KernelParams params;
  int num_full_heads;
  int num_main_groups;
  int heads_last_wave;
  int main_steps;
  int rest_steps;
};
