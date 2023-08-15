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

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "launch_params.h"
#include "bwd_device_gemm_template.h"

namespace bwd_device_gemm {
template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
class DeviceGemmInstanceLauncher {
 public:
  // constructor
  explicit DeviceGemmInstanceLauncher()
    : device_gemm_instance_ptr_(std::make_unique<DeviceGemmTemplate<DeviceGemmTraits>>()) {}

  void Launch(FlashBwdParams &params, hipStream_t &stream);

 private:
  std::unique_ptr<DeviceGemmTemplate<DeviceGemmTraits>> device_gemm_instance_ptr_;
}; // class BwdDeviceGemmInstanceLauncher

template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
void DeviceGemmInstanceLauncher<DeviceGemmTemplate, DeviceGemmTraits>::Launch(FlashBwdParams &params, hipStream_t &stream) {
  bool time_kernel = false;
  bool input_permute = true;
  bool output_permute = true;

  float alpha = params.scale_bmm1f;
  auto seeds = unpack(params.philox_args);

  auto a_element_op     = device_gemm_trait::AElementOp{};
  auto b0_element_op    = device_gemm_trait::B0ElementOp{};
  auto acc0_element_op  = device_gemm_trait::Acc0ElementOp{alpha};
  auto b1_element_op    = device_gemm_trait::B1ElementOp{};
  auto c_element_op     = device_gemm_trait::CElementOp{};

  auto p_q = params.q_ptr;
  auto p_k = params.k_ptr;
  auto p_v = params.v_ptr;
  auto p_y = params.y_ptr;
  auto p_z = params.z_ptr;
  auto p_lse = params.lse_ptr;
  auto p_ygrad = params.ygrad_ptr;
  auto p_qgrad = params.qgrad_ptr;
  auto p_kgrad = params.kgrad_ptr;
  auto p_vgrad = params.vgrad_ptr;
  int batch_size = params.b;
  int num_heads = params.h;
  int head_dim = params.d;
  float dropout_ratio = params.p_dropout;

  using DeviceGemmTemplateInstance = DeviceGemmTemplate<DeviceGemmTraits>;
  std::vector<typename DeviceGemmTemplateInstance::ProblemDesc> problem_descs;

  for (size_t i = 0; i < batch_size; i++) {
    int M = params.host_seqlens_q[i + 1] - params.host_seqlens_q[i]; // seqlen Q
    int N = params.host_seqlens_k[i + 1] - params.host_seqlens_k[i]; // seqlen K
    int K = head_dim;
    int O = head_dim;
    int G0 = 1; // G0 = batch_size
    int G1 = num_heads;

    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1}
            // Q layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K,
                                      1}; // Q layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1}
            // K layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K,
                                      1}; // K layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O}
            // V layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1,
                                      O}; // V layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1}
            // Y layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O,
                                      1}; // Y layout [G0, G1, M, O]

    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides = 
        input_permute
        ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1}
        // Z layout [G0, M, G1, N]
        : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]
    // The softmax stat log-sum-exp (LSE) is used to speed up softmax
    // calculation in backward pass Pi = exp(Si) / sum(exp(S0) + exp(S1) +
    // ...)
    //    = exp(Si) / exp(log(sum(exp() + ...)))
    //    = exp(Si - log(sum(exp() + ...)))
    //               ^^^^^^^^^^^^^^^^^^^^^
    //                       LSE
    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    problem_descs.push_back({
        q_gs_ms_ks_lengths,
        q_gs_ms_ks_strides,
        k_gs_ns_ks_lengths,
        k_gs_ns_ks_strides,
        z_gs_ms_ns_lengths,
        z_gs_ms_ns_strides,
        v_gs_os_ns_lengths,
        v_gs_os_ns_strides,
        y_gs_ms_os_lengths,
        y_gs_ms_os_strides,
        lse_gs_ms_lengths,
        lse_gs_ms_strides,
        {}, // acc0_biases_gs_ms_ns_lengths
        {}, // acc0_biases_gs_ms_ns_strides
        {}, // acc1_biases_gs_ms_os_lengths
        {}  // acc1_biases_gs_ms_os_strides
    });
  }
  // do GEMM
  auto invoker = device_gemm_instance_ptr_->MakeInvoker();

  auto argument = device_gemm_instance_ptr_->MakeArgument(
      p_q, p_k, p_z, p_v, p_y, p_lse, p_ygrad, p_qgrad, p_kgrad, p_vgrad, {},
      {}, problem_descs, a_element_op, b0_element_op, acc0_element_op,
      b1_element_op, c_element_op, dropout_ratio, seeds);

  // specify workspace for problem_desc
  SimpleDeviceMem problem_desc_workspace{device_gemm_instance_ptr_->GetWorkSpaceSize(&argument)};

  device_gemm_instance_ptr_->SetWorkSpacePointer(&argument,
                          problem_desc_workspace.GetDeviceBuffer());

  if (!device_gemm_instance_ptr_->IsSupportedArgument(argument)) {
    std::cout << device_gemm_instance_ptr_->GetTypeString() << " does not support this problem"
              << std::endl;
    return;
  }

  float avg_time = invoker.Run(argument, StreamConfig{stream, time_kernel});

  if (time_kernel) {
    std::cout << "time elpase is " << avg_time << " ms" << std::endl;
  }
} // end of function Launch
} // namespace bwd_device_gemm