// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "fmha.h"
#include "fp16_switch.h"

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using MaskingSpecialization = ck::tensor_operation::device::MaskingSpecialization;

static constexpr auto MaskingSpec_default = 
    MaskingSpecialization::MaskDisabled;
static constexpr auto MaskingSpec_causal =
    MaskingSpecialization::MaskOutUpperTriangle;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;
    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }
    void* GetDeviceBuffer() { return p_mem_; }
    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

template<typename InputType, 
         ck::index_t MPerBlock,    ck::index_t NPerBlock, ck::index_t KPerBlock,   ck::index_t Gemm1NPerBlock, ck::index_t Gemm1KPerBlock,
         ck::index_t MPerXDL,      ck::index_t NPerXDL,   ck::index_t NXdlPerWave, ck::index_t Gemm1NXdlPerWave,
         typename ABlockTransfer,  bool ABlockLdsExtraM,  typename BBlockTransfer, bool B0BlockLdsExtraN,
         typename B1BlockTransfer, ck::index_t B1BlockTransferSrcScalarPerVector, 
         ck::index_t CShuffleNXdlPerWavePerShuffle, typename CShuffleBlockTransferClusterLengths, 
         MaskingSpecialization MaskingSpec>
void run_fmha_fp16_bf16_gfx90a_loop_(LaunchParams<FmhaFpropParams> &launch_params){
    using F32 = float;
    using INT32 = int;
    using BF16 = ck::bhalf_t;
    using FP16 = ck::half_t;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ADataType        = InputType;
    using B0DataType       = InputType;
    using B1DataType       = InputType;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using CDataType        = InputType;
    using GemmDataType     = InputType;
    using ZDataType        = INT32;
    using LSEDataType      = F32;
    using Acc0BiasDataType = ck::Tuple<>;
    using Acc1BiasDataType = ck::Tuple<>;

    static constexpr ck::index_t NumDimG = 2;
    static constexpr ck::index_t NumDimM = 1;
    static constexpr ck::index_t NumDimN = 1;
    static constexpr ck::index_t NumDimK = 1;
    static constexpr ck::index_t NumDimO = 1;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

    static constexpr auto TensorSpecA  = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB0 = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB1 = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecC  = ck::tensor_operation::device::TensorSpecialization::Default;

    static constexpr bool deterministic = true;
    static constexpr bool nondeterministic = false;
    
    bool is_deterministic = launch_params.params.is_deterministic;

    //init the instance with parameters
    using DeviceGemmInstance1 =
        ck::tensor_operation::device::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle<
            NumDimG,
            NumDimM,
            NumDimN,
            NumDimK,
            NumDimO,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            GemmDataType,
            ZDataType,
            LSEDataType,
            Acc0BiasDataType,
            Acc1BiasDataType,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            TensorSpecA,
            TensorSpecB0,
            TensorSpecB1,
            TensorSpecC,
            1,
            256,
            MPerBlock,         // MPerBlock
            NPerBlock,         // NPerBlock
            KPerBlock,         // KPerBlock
            Gemm1NPerBlock,    // Gemm1NPerBlock
            Gemm1KPerBlock,    // Gemm1KPerBlock
            8,                 // AK1
            8,                 // BK1
            2,                 // B1K1
            MPerXDL,           // MPerXDL
            NPerXDL,           // NPerXDL
            1,                 // MXdlPerWave
            NXdlPerWave,       // NXdlPerWave
            Gemm1NXdlPerWave,  // Gemm1NXdlPerWave
            ABlockTransfer,    // ABlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            ABlockLdsExtraM,   // ABlockLdsExtraM
            BBlockTransfer,    // BBlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            B0BlockLdsExtraN,  // B0BlockLdsExtraN
            B1BlockTransfer,   // B1BlockTransfer
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            B1BlockTransferSrcScalarPerVector,    //B1BlockTransferSrcScalarPerVector
            2,
            false,
            1,                                    // CShuffleMXdlPerWavePerShuffle
            CShuffleNXdlPerWavePerShuffle,        // CShuffleNXdlPerWavePerShuffle
            CShuffleBlockTransferClusterLengths,  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,                                    // CShuffleBlockTransferScalarPerVector_NPerBlock
            MaskingSpec,
            deterministic>;                       // MaskingSpecialization

    using DeviceGemmInstance2 =
        ck::tensor_operation::device::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle<
            NumDimG,
            NumDimM,
            NumDimN,
            NumDimK,
            NumDimO,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            GemmDataType,
            ZDataType,
            LSEDataType,
            Acc0BiasDataType,
            Acc1BiasDataType,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            TensorSpecA,
            TensorSpecB0,
            TensorSpecB1,
            TensorSpecC,
            1,
            256,
            MPerBlock,         // MPerBlock
            NPerBlock,         // NPerBlock
            KPerBlock,         // KPerBlock
            Gemm1NPerBlock,    // Gemm1NPerBlock
            Gemm1KPerBlock,    // Gemm1KPerBlock
            8,                 // AK1
            8,                 // BK1
            2,                 // B1K1
            MPerXDL,           // MPerXDL
            NPerXDL,           // NPerXDL
            1,                 // MXdlPerWave
            NXdlPerWave,       // NXdlPerWave
            Gemm1NXdlPerWave,  // Gemm1NXdlPerWave
            ABlockTransfer,    // ABlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            ABlockLdsExtraM,   // ABlockLdsExtraM
            BBlockTransfer,    // BBlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            B0BlockLdsExtraN,  // B0BlockLdsExtraN
            B1BlockTransfer,   // B1BlockTransfer
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            B1BlockTransferSrcScalarPerVector,    //B1BlockTransferSrcScalarPerVector
            2,
            false,
            1,                                    // CShuffleMXdlPerWavePerShuffle
            CShuffleNXdlPerWavePerShuffle,        // CShuffleNXdlPerWavePerShuffle
            CShuffleBlockTransferClusterLengths,  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,                                    // CShuffleBlockTransferScalarPerVector_NPerBlock
            MaskingSpec,
            nondeterministic>;                       // MaskingSpecialization
        
    bool time_kernel    = false;

    bool input_permute = true;
    bool output_permute = true;

    bool z_tensor_permute = false;

    float alpha = launch_params.params.scale_bmm1f;

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    auto p_a = launch_params.params.q_ptr;
    auto p_b0 = launch_params.params.k_ptr;
    auto p_b1 = launch_params.params.v_ptr;
    auto p_c = launch_params.params.o_ptr;
    auto p_z = launch_params.params.s_ptr;
    auto p_lse = launch_params.params.softmax_lse_ptr;

    if (is_deterministic) {
      std::vector<typename DeviceGemmInstance1::ProblemDesc> problem_descs;

      int batch_size = launch_params.params.b;
      int num_heads = launch_params.params.h;
      int head_dim = launch_params.params.d;

      float dropout_ratio = launch_params.params.p_dropout;

      auto seeds = unpack(launch_params.params.philox_args);

      auto seed_   = std::get<0>(seeds);
      auto offset_ = std::get<1>(seeds);

      //std::cout << "fwd seed is " << seed_ ;
      //std::cout << " , fwd offset is " << offset_ << std::endl;

      for(size_t i = 0; i < batch_size ; i++){
          int M     = launch_params.params.host_seqlens_q[i + 1] - launch_params.params.host_seqlens_q[i]; //seqlen Q
          int N     = launch_params.params.host_seqlens_k[i + 1] - launch_params.params.host_seqlens_k[i]; //seqlen K
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
      auto gemm     = DeviceGemmInstance1{};
      auto invoker  = gemm.MakeInvoker();
      auto argument = gemm.MakeArgument(p_a,
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
      SimpleDeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

      gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

      if(!gemm.IsSupportedArgument(argument))
      {
          std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

          return;
      }

      float ave_time = invoker.Run(argument, StreamConfig{launch_params.stream, time_kernel});

      if(time_kernel){
          std::cout << "time elpase is " << ave_time <<" ms" << std::endl;
      }
    } else {
      std::vector<typename DeviceGemmInstance2::ProblemDesc> problem_descs;

      int batch_size = launch_params.params.b;
      int num_heads = launch_params.params.h;
      int head_dim = launch_params.params.d;

      float dropout_ratio = launch_params.params.p_dropout;

      auto seeds = unpack(launch_params.params.philox_args);

      auto seed_   = std::get<0>(seeds);
      auto offset_ = std::get<1>(seeds);

      //std::cout << "fwd seed is " << seed_ ;
      //std::cout << " , fwd offset is " << offset_ << std::endl;

      for(size_t i = 0; i < batch_size ; i++){
          int M     = launch_params.params.host_seqlens_q[i + 1] - launch_params.params.host_seqlens_q[i]; //seqlen Q
          int N     = launch_params.params.host_seqlens_k[i + 1] - launch_params.params.host_seqlens_k[i]; //seqlen K
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
      auto gemm     = DeviceGemmInstance2{};
      auto invoker  = gemm.MakeInvoker();
      auto argument = gemm.MakeArgument(p_a,
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
      SimpleDeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

      gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

      if(!gemm.IsSupportedArgument(argument))
      {
          std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

          return;
      }

      float ave_time = invoker.Run(argument, StreamConfig{launch_params.stream, time_kernel});

      if(time_kernel){
          std::cout << "time elpase is " << ave_time <<" ms" << std::endl;
      }
    }
}


void run_fmha_fp16_bf16_gfx90a(LaunchParams<FmhaFpropParams> &launch_params) {

    //template<typename InputType, 
    //ck::index_t MPerBlock,    ck::index_t NPerBlock, ck::index_t KPerBlock,   ck::index_t Gemm1NPerBlock, ck::index_t Gemm1KPerBlock,
    //ck::index_t MPerXDL,      ck::index_t NPerXDL,   ck::index_t NXdlPerWave, ck::index_t Gemm1NXdlPerWave,
    //typename ABlockTransfer,  bool ABlockLdsExtraM,  typename BBlockTransfer, bool B0BlockLdsExtraN,
    //typename B1BlockTransfer, ck::index_t B1BlockTransferSrcScalarPerVector, 
    //ck::index_t CShuffleNXdlPerWavePerShuffle, typename CShuffleBlockTransferClusterLengths, 
    //MaskingSpecialization MaskingSpec>

    FP16_SWITCH(launch_params.params.is_bf16, [&] {
        if(launch_params.params.is_causal){
            if(launch_params.params.d <= 32){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 32, 32,
                                                            32,  32,  4,  1, 
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<16, 16, 1>, 2, 
                                                            1, S<1, 64, 1, 4>,
                                                            MaskingSpec_causal>(launch_params);
            }
            else if(launch_params.params.d <= 64){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 64, 32,
                                                            32,  32,  4,  2,
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<16, 16, 1>, 4, 
                                                            2, S<1, 32, 1, 8>,
                                                            MaskingSpec_causal>(launch_params);
            }
            else if(launch_params.params.d <= 128){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 128, 32,
                                                            32,  32,  4,  4, 
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<8, 32, 1>, 4, 
                                                            2, S<1, 32, 1, 8>,
                                                            MaskingSpec_causal>(launch_params);

            }
        }
        else{
            if(launch_params.params.d <= 32){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 32, 32,
                                                            32,  32,  4,  1, 
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<16, 16, 1>, 2, 
                                                            1, S<1, 64, 1, 4>,
                                                            MaskingSpec_default>(launch_params);
            }
            else if(launch_params.params.d <= 64){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 64, 32,
                                                            32,  32,  4,  2, 
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<16, 16, 1>, 4, 
                                                            2, S<1, 32, 1, 8>,
                                                            MaskingSpec_default>(launch_params);
            }
            else if(launch_params.params.d <= 128){
                run_fmha_fp16_bf16_gfx90a_loop_<elem_type,  128, 128, 32, 128, 32,
                                                            32,  32,  4,  4, 
                                                            S<4, 64, 1>, true, S<4, 64, 1>, true,
                                                            S<8, 32, 1>, 4, 
                                                            2, S<1, 32, 1, 8>,
                                                            MaskingSpec_default>(launch_params);
            }
        }
    });

}