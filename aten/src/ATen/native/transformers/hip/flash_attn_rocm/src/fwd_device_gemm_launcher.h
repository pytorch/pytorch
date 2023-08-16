// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <iostream>

#include <ATen/native/transformers/hip/flash_attn_rocm/src/launch_params.h>
#include <ATen/native/transformers/hip/flash_attn_rocm/src/fwd_device_gemm_template.h>

namespace fwd_device_gemm {
template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
class DeviceGemmInstanceLauncher {
 public:
  // constructor
  explicit DeviceGemmInstanceLauncher()
    : device_gemm_instance_ptr_(std::make_unique<DeviceGemmTemplate<DeviceGemmTraits>>()) {}
  
  void Launch(FlashFwdParams &params, hipStream_t &stream);

 private:
  std::unique_ptr<DeviceGemmTemplate<DeviceGemmTraits>> device_gemm_instance_ptr_;
}; // class DeviceGemmInstanceLauncher

template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
void DeviceGemmInstanceLauncher<DeviceGemmTemplate, DeviceGemmTraits>::Launch(FlashFwdParams &params, hipStream_t &stream) {
  bool time_kernel = false;
  bool input_permute = true;
  bool output_permute = true;
  bool z_tensor_permute = false;

  float alpha = params.scale_bmm1f;

  auto a_element_op     = device_gemm_trait::AElementOp{};
  auto b0_element_op    = device_gemm_trait::B0ElementOp{};
  auto acc0_element_op  = device_gemm_trait::Acc0ElementOp{alpha};
  auto b1_element_op    = device_gemm_trait::B1ElementOp{};
  auto c_element_op     = device_gemm_trait::CElementOp{};

  auto p_a = params.q_ptr;
  auto p_b0 = params.k_ptr;
  auto p_b1 = params.v_ptr;
  auto p_c = params.o_ptr;
  auto p_z = params.s_ptr;
  auto p_lse = params.softmax_lse_ptr;

  std::vector<typename DeviceGemmTemplate<DeviceGemmTraits>::ProblemDesc> problem_descs;

  int batch_size = params.b;
  int num_heads = params.h;
  int head_dim = params.d;

  float dropout_ratio = params.p_dropout;

  auto seeds = unpack(params.philox_args);

  auto seed_   = std::get<0>(seeds);
  auto offset_ = std::get<1>(seeds);

  //std::cout << "fwd seed is " << seed_ ;
  //std::cout << " , fwd offset is " << offset_ << std::endl;

  for(size_t i = 0; i < batch_size ; i++){
    int M     = params.host_seqlens_q[i + 1] - params.host_seqlens_q[i]; //seqlen Q
    int N     = params.host_seqlens_k[i + 1] - params.host_seqlens_k[i]; //seqlen K
    int K     = head_dim;
    int O     = head_dim;
    int G0 = 1; // G0 = batch_size
    int G1 = num_heads;
    

    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // B0 layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // B1 layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]
    
    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides =
        z_tensor_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]

    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides =
        std::vector<ck::index_t>{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    problem_descs.push_back({a_gs_ms_ks_lengths,
                            a_gs_ms_ks_strides,
                            b0_gs_ns_ks_lengths,
                            b0_gs_ns_ks_strides,
                            b1_gs_os_ns_lengths,
                            b1_gs_os_ns_strides,
                            c_gs_ms_os_lengths,
                            c_gs_ms_os_strides,
                            z_gs_ms_ns_lengths,
                            z_gs_ms_ns_strides,
                            lse_gs_ms_lengths,
                            lse_gs_ms_strides,
                            {},   // acc0_biases_gs_ms_ns_lengths
                            {},   // acc0_biases_gs_ms_ns_strides
                            {},   // acc1_biases_gs_ms_os_lengths
                            {}}); // acc1_biases_gs_ms_os_strides
                              
  }

  // do GEMM
  auto invoker  = device_gemm_instance_ptr_->MakeInvoker();
  auto argument = device_gemm_instance_ptr_->MakeArgument(
      p_a,
      p_b0,
      p_b1,
      p_c,
      p_z,
      p_lse,
      {},
      {},
      problem_descs,
      a_element_op,
      b0_element_op,
      acc0_element_op,
      b1_element_op,
      c_element_op,
      dropout_ratio,
      seeds);

  // specify workspace for problem_desc
  SimpleDeviceMem problem_desc_workspace{device_gemm_instance_ptr_->GetWorkSpaceSize(&argument)};

  device_gemm_instance_ptr_->SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

  if(!device_gemm_instance_ptr_->IsSupportedArgument(argument))
  {
      std::cout << device_gemm_instance_ptr_->GetTypeString() << " does not support this problem" << std::endl;

      return;
  }

  float avg_time = invoker.Run(argument, StreamConfig{stream, time_kernel});

  if(time_kernel){
      std::cout << "time elpase is " << avg_time <<" ms" << std::endl;
  }
}
} // namespace fwd_device_gemm