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

#include "fmha.h"
#include "fp16_switch.h"

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

template <ck::index_t... Is> using S = ck::Sequence<Is...>;
using MaskingSpecialization = ck::tensor_operation::device::MaskingSpecialization;

static constexpr auto kMaskingSpecializationDefault = MaskingSpecialization::MaskDisabled;
static constexpr auto kMaskingSpecializationCausal = MaskingSpecialization::MaskOutUpperTriangle;

struct SimpleDeviceMem {
  SimpleDeviceMem() = delete;
  SimpleDeviceMem(std::size_t mem_size) : p_mem_{} {
    (void)hipMalloc(static_cast<void **>(&p_mem_), mem_size);
  }
  void *GetDeviceBuffer() { return p_mem_; }
  ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

  void *p_mem_;
};

template <typename InputType, 
          typename OutputType, 
          typename DropoutType,
          typename GemmDataType,
          ck::index_t version, 
          ck::index_t c_shuffle_block_transfer_scalar_per_vector_n_per_block,
          MaskingSpecialization masking_specialization>
void run_fmha_dgrad_fp16_bf16_gfx90a_loop_(LaunchParams<FmhaDgradParams> &launch_params) {
  using Int32 = int;
  using Int16 = unsigned short;
  using Float32 = float;
  using BFloat16 = ck::bhalf_t;
  using Float16 = ck::half_t;

  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using Scale = ck::tensor_operation::element_wise::Scale;

  using QkvElementOp = PassThrough;
  using YElementOp = PassThrough;

  using InputDataType    = InputType;
  using OutputDataType   = OutputType;
  using AccDataType      = Float32;
  using ShuffleDataType  = Float32;
  using LSEDataType      = Float32;
  using ZDataType        = DropoutType;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;

  static constexpr ck::index_t NumDimG = 2;
  static constexpr ck::index_t NumDimM = 1;
  static constexpr ck::index_t NumDimN = 1;
  static constexpr ck::index_t NumDimK = 1;
  static constexpr ck::index_t NumDimO = 1;

  static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  
  static constexpr auto TensorSpecQ = ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecK = ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecV = ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecY = ck::tensor_operation::device::TensorSpecialization::Default;

  static constexpr bool deterministic = true;
  static constexpr bool nondeterministic = false;

  bool is_deterministic = launch_params.params.is_deterministic;
  bool time_kernel = false;
  bool input_permute = true;
  bool output_permute = true;

  float alpha = launch_params.params.scale_bmm1f;
  auto seeds = unpack(launch_params.params.philox_args);

  auto seed_   = std::get<0>(seeds);
  auto offset_ = std::get<1>(seeds);

  //std::cout << "bwd seed is " << seed_ ;
  //std::cout << " , bwd offset is " << offset_ << std::endl;

  auto a_element_op = QkvElementOp{};
  auto b0_element_op = QkvElementOp{};
  auto acc0_element_op = Scale{alpha};
  auto b1_element_op = QkvElementOp{};
  auto c_element_op = YElementOp{};

  auto p_q = launch_params.params.q_ptr;
  auto p_k = launch_params.params.k_ptr;
  auto p_v = launch_params.params.v_ptr;
  auto p_y = launch_params.params.y_ptr;
  auto p_z = launch_params.params.z_ptr;
  auto p_lse = launch_params.params.lse_ptr;
  auto p_ygrad = launch_params.params.ygrad_ptr;
  auto p_qgrad = launch_params.params.qgrad_ptr;
  auto p_kgrad = launch_params.params.kgrad_ptr;
  auto p_vgrad = launch_params.params.vgrad_ptr;
  int batch_size = launch_params.params.b;
  int num_heads = launch_params.params.h;
  int head_dim = launch_params.params.d;
  float dropout_ratio = launch_params.params.p_dropout;
  // init the instance with parameters
  auto run_kernel = [&]<typename DeviceGemmInstance>(DeviceGemmInstance gemm) {
    std::vector<typename DeviceGemmInstance::ProblemDesc> problem_descs;
    for (size_t i = 0; i < batch_size; i++) {
      int M = launch_params.params.host_seqlens_q[i + 1] -
              launch_params.params.host_seqlens_q[i]; // seqlen Q
      int N = launch_params.params.host_seqlens_k[i + 1] -
              launch_params.params.host_seqlens_k[i]; // seqlen K
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
              : std::vector<ck::index_t>{G1 * M * N, M * N, N,
                                         1}; // Z layout [G0, G1, M, N]
      // The softmax stat log-sum-exp (LSE) is used to speed up softmax
      // calculation in backward pass Pi = exp(Si) / sum(exp(S0) + exp(S1) +
      // ...)
      //    = exp(Si) / exp(log(sum(exp() + ...)))
      //    = exp(Si - log(sum(exp() + ...)))
      //               ^^^^^^^^^^^^^^^^^^^^^
      //                       LSE
      std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
      std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M,
                                                 1}; // LSE layout [G0, G1, M]

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
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(
        p_q, p_k, p_z, p_v, p_y, p_lse, p_ygrad, p_qgrad, p_kgrad, p_vgrad, {},
        {}, problem_descs, a_element_op, b0_element_op, acc0_element_op,
        b1_element_op, c_element_op, dropout_ratio, seeds);

    // specify workspace for problem_desc
    SimpleDeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument,
                             problem_desc_workspace.GetDeviceBuffer());

    if (!gemm.IsSupportedArgument(argument)) {
      std::cout << gemm.GetTypeString() << " does not support this problem"
                << std::endl;
      return;
    }

    float ave_time = invoker.Run(argument, StreamConfig{launch_params.stream, time_kernel});

    if (time_kernel) {
      std::cout << "time elpase is " << ave_time << " ms" << std::endl;
    }
  };
  // deterministic mode
  if (is_deterministic) {
    if (version == 1) {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          64,          // KPerBlock
          128,         // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          4,           // Gemm1NXdlPerWave
          2,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          4, // CShuffleNXdlPerWavePerShuffle
          S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          deterministic>;
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else if (version == 2) {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          64,          // KPerBlock
          64,          // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          2,           // Gemm1NXdlPerWave
          2,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          2, // CShuffleNXdlPerWavePerShuffle
          S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          deterministic>; 
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          32,          // KPerBlock
          32,          // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          1,           // Gemm1NXdlPerWave
          1,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          1, // CShuffleNXdlPerWavePerShuffle
          S<1, 64, 1, 4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          deterministic>; 
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    }
  // non-deterministic mode
  } else {
    if (version == 1) {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          64,          // KPerBlock
          128,         // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          4,           // Gemm1NXdlPerWave
          2,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          4, // CShuffleNXdlPerWavePerShuffle
          S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          nondeterministic>;
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else if (version == 2) {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          64,          // KPerBlock
          64,          // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          2,           // Gemm1NXdlPerWave
          2,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          2, // CShuffleNXdlPerWavePerShuffle
          S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          nondeterministic>; 
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else {
      using DeviceGemmInstance = ck::tensor_operation::device::
        DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, InputDataType, OutputDataType, GemmDataType,
          ZDataType, LSEDataType, Acc0BiasDataType, Acc1BiasDataType,
          AccDataType, ShuffleDataType, QkvElementOp, QkvElementOp, Scale,
          QkvElementOp, YElementOp, GemmSpec, TensorSpecQ, TensorSpecK,
          TensorSpecV, TensorSpecY, 1, 256,
          128,         // MPerBlock
          128,         // NPerBlock
          32,          // KPerBlock
          32,          // Gemm1NPerBlock
          32,          // Gemm1KPerBlock
          8,           // AK1
          8,           // BK1
          2,           // B1K1
          32,          // MPerXDL
          32,          // NPerXDL
          1,           // MXdlPerWave
          4,           // NXdlPerWave
          1,           // Gemm1NXdlPerWave
          1,           // Gemm2NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
          S<8, 32, 1>, // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1, // CShuffleMXdlPerWavePerShuffle
          1, // CShuffleNXdlPerWavePerShuffle
          S<1, 64, 1, 4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_specialization, // MaskingSpecialization
          nondeterministic>; 
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    }
  }
}

void run_fmha_dgrad_fp16_bf16_gfx90a(LaunchParams<FmhaDgradParams> &launch_params) {
  using Int32 = int;
  using Int16 = unsigned short;
  using Float32 = float;
  using Float16 = ck::half_t;
  using BFloat16 = ck::bhalf_t;

  if (launch_params.params.is_performance_mode) {
    if (launch_params.params.is_bf16) {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 1, 8, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 2, 8, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 3, 8, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 1, 8, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 2, 8, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, BFloat16, Int32, BFloat16, 3, 8, kMaskingSpecializationDefault>(launch_params);
        }
      }
    } 
    else {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 1, 8, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 2, 8, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 3, 8, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 1, 8, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 2, 8, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float16, Int16, BFloat16, 3, 8, kMaskingSpecializationDefault>(launch_params);
        }
      }
    }
  // non-performance mode
  } else {
    if (launch_params.params.is_bf16) {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 1, 4, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 2, 4, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 3, 4, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 1, 4, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 2, 4, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<BFloat16, Float32, Int32, BFloat16, 3, 4, kMaskingSpecializationDefault>(launch_params);
        }
      }
    } else {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 1, 4, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 2, 4, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 3, 4, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 1, 4, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 2, 4, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<Float16, Float32, Int16, Float16, 3, 4, kMaskingSpecializationDefault>(launch_params);
        }
      }
    }
  }
}