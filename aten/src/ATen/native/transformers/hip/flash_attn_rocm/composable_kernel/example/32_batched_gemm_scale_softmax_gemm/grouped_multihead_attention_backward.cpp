// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
/*
Backprop for Gemm + Softmax + Gemm fused operation, where forward prop is defined as:

  Y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o

Computation graph:

          K^T                   V
          |                     |
          |                     |
    Q --- * ----- Softmax ----- * --> Y
              S             P

Kernel inputs:

    Q, K, V, Y, dY, per-row softmax stats (LSE)

Kernel outputs:

    dQ, dK, dV

*/

#define USING_MASK 0
#define DIM 64 // DIM should be a multiple of 8.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <fstream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_backward_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_backward_xdl_cshuffle_v2.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_dropout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using U16   = unsigned short;
using INT32 = int32_t;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using QKVElementOp = PassThrough;
using YElementOp   = PassThrough;

using InputDataType    = BF16;
using OutputDataType   = F32;
using GemmDataType     = BF16;
using AccDataType      = F32;
using ShuffleDataType  = F32;
using LSEDataType      = F32;
using ZDataType        = INT32; // U16
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;
// When OutputDataType == F32,      CShuffleBlockTransferScalarPerVector_NPerBlock = 4
// When OutputDataType == F16/BF16, CShuffleBlockTransferScalarPerVector_NPerBlock = 8
static constexpr ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock = 4;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
#if USING_MASK
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle;
#else
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
#endif

static constexpr auto TensorSpecQ   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecK   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecV   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecY   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr bool Deterministic = true;

// DIM should be a multiple of 8.
// If      DIM <= 32 , ues prototype1 1st template.
// If 32 < DIM <= 64 , ues prototype1 2nd template.
// If 64 < DIM <= 128, ues prototype2 2nd template.
#if(DIM <= 32)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        OutputDataType,
        GemmDataType,
        ZDataType,
        LSEDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        ShuffleDataType,
        QKVElementOp,
        QKVElementOp,
        Scale,
        QKVElementOp,
        YElementOp,
        GemmSpec,
        TensorSpecQ,
        TensorSpecK,
        TensorSpecV,
        TensorSpecY,
        1,
        256,
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
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        1,              // CShuffleNXdlPerWavePerShuffle
        S<1, 64, 1, 4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        CShuffleBlockTransferScalarPerVector_NPerBlock, // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,                                    // MaskingSpecialization
        Deterministic>;
#elif(DIM <= 64)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        OutputDataType,
        GemmDataType,
        ZDataType,
        LSEDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        ShuffleDataType,
        QKVElementOp,
        QKVElementOp,
        Scale,
        QKVElementOp,
        YElementOp,
        GemmSpec,
        TensorSpecQ,
        TensorSpecK,
        TensorSpecV,
        TensorSpecY,
        1,
        256,
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
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        CShuffleBlockTransferScalarPerVector_NPerBlock, // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,                                    // MaskingSpecialization
        Deterministic>;

// using DeviceGemmInstance =
//     ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
//         NumDimG,
//         NumDimM,
//         NumDimN,
//         NumDimK,
//         NumDimO,
//         InputDataType,
//         OutputDataType,
//         GemmDataType,
//         ZDataType,
//         LSEDataType,
//         Acc0BiasDataType,
//         Acc1BiasDataType,
//         AccDataType,
//         ShuffleDataType,
//         QKVElementOp,
//         QKVElementOp,
//         Scale,
//         QKVElementOp,
//         YElementOp,
//         GemmSpec,
//         TensorSpecQ,
//         TensorSpecK,
//         TensorSpecV,
//         TensorSpecY,
//         1,
//         256,
//         128,         // MPerBlock
//         128,         // NPerBlock
//         64,          // KPerBlock
//         64,          // Gemm1NPerBlock
//         64,          // Gemm1KPerBlock
//         8,           // AK1
//         8,           // BK1
//         2,           // B1K1
//         32,          // MPerXDL
//         32,          // NPerXDL
//         1,           // MXdlPerWave
//         4,           // NXdlPerWave
//         2,           // Gemm1NXdlPerWave
//         2,           // Gemm2NXdlPerWave
//         S<4, 64, 1>, // ABlockTransfer
//         S<1, 0, 2>,
//         S<1, 0, 2>,
//         2,
//         8,
//         8,
//         true,
//         S<4, 64, 1>, // BBlockTransfer
//         S<1, 0, 2>,
//         S<1, 0, 2>,
//         2,
//         8,
//         8,
//         true,
//         S<8, 32, 1>, // B1BlockTransfer
//         S<0, 2, 1>,
//         S<0, 2, 1>,
//         1,
//         2,
//         2,
//         false,
//         1,              // CShuffleMXdlPerWavePerShuffle
//         2,              // CShuffleNXdlPerWavePerShuffle
//         S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
//         CShuffleBlockTransferScalarPerVector_NPerBlock,
//         MaskingSpec,
//         Deterministic>;
#elif(DIM <= 128)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        OutputDataType,
        GemmDataType,
        ZDataType,
        LSEDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        ShuffleDataType,
        QKVElementOp,
        QKVElementOp,
        Scale,
        QKVElementOp,
        YElementOp,
        GemmSpec,
        TensorSpecQ,
        TensorSpecK,
        TensorSpecV,
        TensorSpecY,
        1,
        256,
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
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        4,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        CShuffleBlockTransferScalarPerVector_NPerBlock, // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,                                    // MaskingSpecialization
        Deterministic>;
#endif

// Ref Gemm0: S = alpha * Q * K^T
// fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<InputDataType,
                                                                                InputDataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                Scale>;

// Ref Softmax: P = Softmax(S)
// fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, InputDataType, AccDataType>;

// Ref Gemm1: Y = P * V
// fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<InputDataType,
                                                                                InputDataType,
                                                                                InputDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

// Ref Gemm for backward pass
// fp16 in, fp16 out
using ReferenceGemm0GradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<InputDataType,
                                                                                    InputDataType,
                                                                                    InputDataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;

using ReferenceGemm1GradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<InputDataType,
                                                                                    InputDataType,
                                                                                    OutputDataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;

// Ref dropout
using ReferenceDropoutInstance =
    ck::tensor_operation::host::ReferenceDropout<ZDataType, InputDataType, InputDataType>;

template <typename TensorQ,
          typename TensorK,
          typename TensorV,
          typename TensorS,
          typename TensorP,
          typename TensorZ,
          typename TensorY,
          typename TensorLSE = TensorP>
void run_attention_fwd_host(const TensorQ& q_g_m_k,
                            const TensorK& k_g_n_k,
                            const TensorV& v_g_n_o,
                            const float alpha,
                            TensorS& s_g_m_n,
                            TensorP& p_g_m_n,
                            TensorY& y_g_m_o,
                            TensorLSE& lse_g_m,
                            TensorP& p_drop_g_m_n,
                            TensorZ& z_g_m_n,
                            ZDataType p_dropout_in_16bits,
                            float rp_dropout)
{
    // S = alpha * Q * K^T
    auto k_g_k_n            = k_g_n_k.Transpose({0, 2, 1});
    auto ref_gemm0          = ReferenceGemm0Instance{};
    auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
    auto ref_gemm0_argument = ref_gemm0.MakeArgument(
        q_g_m_k, k_g_k_n, s_g_m_n, PassThrough{}, PassThrough{}, Scale{alpha});

    ref_gemm0_invoker.Run(ref_gemm0_argument);

    // masking
    auto N          = s_g_m_n.GetLengths()[2];
    const auto mask = DeviceGemmInstance::C0MatrixMask(N);
    s_g_m_n.ForEach([&](auto& self, auto idx) {
        if(mask.IsMaskedElement(idx[1], idx[2]))
            self(idx) = -ck::NumericLimits<float>::Infinity();
    });

    // P = Softmax(S)
    auto ref_softmax          = ReferenceSoftmaxInstance{};
    auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
    auto ref_softmax_argument = ref_softmax.MakeArgument(s_g_m_n, p_g_m_n, 1, 0, {2}, &lse_g_m);

    ref_softmax_invoker.Run(ref_softmax_argument);

    // P_dropped
    auto ref_dropout         = ReferenceDropoutInstance{};
    auto ref_dropout_invoker = ref_dropout.MakeInvoker();
    auto ref_dropout_argment =
        ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_16bits, rp_dropout);
    ref_dropout_invoker.Run(ref_dropout_argment);

    // Y = P_dropout * V
    auto ref_gemm1          = ReferenceGemm1Instance{};
    auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
    auto ref_gemm1_argument = ref_gemm1.MakeArgument(
        p_drop_g_m_n, v_g_n_o, y_g_m_o, PassThrough{}, PassThrough{}, PassThrough{});

    ref_gemm1_invoker.Run(ref_gemm1_argument);
}

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 2; // method 1 will have slightly higher error; TODO: to investigate
    bool time_kernel     = true;

    // Overall QKV matrices shape
    // y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o
    // y_g0_g1_m_o = reshape(y_g_m_o, [G0, G1, M, O])
    // y_g0_m_g1_o = permute(y_g0_g1_m_o, [0, 2, 1, 3])
    float alpha  = 1.f / std::sqrt(DIM);
    float p_drop = 0.2;

    bool input_permute  = true;
    bool output_permute = true;

    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 7)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        p_drop = std::stof(argv[4]);

        input_permute  = std::stoi(argv[5]);
        output_permute = std::stoi(argv[6]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    float p_dropout               = 1 - p_drop;
    ZDataType p_dropout_in_16bits = ZDataType(std::floor(p_dropout * 65535.0));
    float rp_dropout              = 1.0 / p_dropout;

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<const void*> p_q;
    std::vector<const void*> p_k;
    std::vector<void*> p_z;         // for result verification
    std::vector<void*> p_z_nullptr; // for time test
    std::vector<const void*> p_v;
    std::vector<const void*> p_y;
    std::vector<const void*> p_lse;
    std::vector<void*> p_qgrad;
    std::vector<void*> p_kgrad;
    std::vector<void*> p_vgrad;
    std::vector<const void*> p_ygrad;

    std::vector<Tensor<InputDataType>> q_g_m_ks;
    std::vector<Tensor<InputDataType>> k_g_n_ks;
    std::vector<Tensor<ZDataType>> z_g_m_ns;
    std::vector<Tensor<InputDataType>> v_g_n_os;
    std::vector<Tensor<AccDataType>> s_g_m_ns;
    std::vector<Tensor<InputDataType>> p_g_m_ns;
    std::vector<Tensor<InputDataType>> y_g_m_os;
    std::vector<Tensor<LSEDataType>> lse_g_ms;
    std::vector<Tensor<InputDataType>> p_drop_g_m_ns;

    std::vector<Tensor<InputDataType>> q_tensors;
    std::vector<Tensor<InputDataType>> k_tensors;
    std::vector<Tensor<InputDataType>> v_tensors;
    std::vector<Tensor<InputDataType>> y_tensors;
    std::vector<Tensor<ZDataType>> z_tensors;
    std::vector<Tensor<LSEDataType>> lse_tensors;
    std::vector<Tensor<OutputDataType>> qgrad_tensors;
    std::vector<Tensor<OutputDataType>> kgrad_tensors;
    std::vector<Tensor<OutputDataType>> vgrad_tensors;
    std::vector<Tensor<InputDataType>> ygrad_tensors;

    std::vector<DeviceMemPtr> q_tensors_device;
    std::vector<DeviceMemPtr> k_tensors_device;
    std::vector<DeviceMemPtr> z_tensors_device;
    std::vector<DeviceMemPtr> v_tensors_device;
    std::vector<DeviceMemPtr> y_tensors_device;
    std::vector<DeviceMemPtr> lse_tensors_device;
    std::vector<DeviceMemPtr> qgrad_tensors_device;
    std::vector<DeviceMemPtr> ygrad_tensors_device;
    std::vector<DeviceMemPtr> kgrad_tensors_device;
    std::vector<DeviceMemPtr> vgrad_tensors_device;
    std::size_t group_count = 10;
    std::size_t flop = 0, num_byte = 0;
    for(std::size_t i = 0; i < group_count; i++)
    {
        int M  = 128 * (rand() % 8) + (rand() % 128);
        int N  = 128 * (rand() % 8) + (rand() % 128);
        int K  = DIM;
        int O  = DIM;
        int G0 = rand() % 4 + 1;
        int G1 = rand() % 4 + 1;
        std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> q_gs_ms_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
                : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

        std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> k_gs_ns_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
                : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

        std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> v_gs_os_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
                : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

        std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> y_gs_ms_os_strides =
            output_permute
                ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
                : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

        std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
        std::vector<ck::index_t> z_gs_ms_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
                : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]
        // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward
        // pass Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
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
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
        });

        int BatchCount = G0 * G1;
        flop += (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;
        // Q/K/V/Y, dQ/dK/dV/dY, LSE
        num_byte += (sizeof(InputDataType) * M * K + sizeof(InputDataType) * K * N +
                     sizeof(InputDataType) * N * O + sizeof(InputDataType) * M * O * size_t(2) +
                     sizeof(OutputDataType) * M * K + sizeof(OutputDataType) * K * N +
                     sizeof(OutputDataType) * N * O) *
                        BatchCount +
                    sizeof(LSEDataType) * M * BatchCount;

        Tensor<InputDataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
        Tensor<InputDataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
        Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
        Tensor<InputDataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
        Tensor<InputDataType> y_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
        Tensor<InputDataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
        Tensor<LSEDataType> lse_gs_ms(lse_gs_ms_lengths, lse_gs_ms_strides);
        if(i < 4)
        {
            std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
            std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
            std::cout << "z_gs_ms_ns: " << z_gs_ms_ns.mDesc << std::endl;
            std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;
            std::cout << "y_gs_ms_os: " << y_gs_ms_os.mDesc << std::endl;
            std::cout << "lse_gs_ms_os: " << lse_gs_ms.mDesc << std::endl;
        }
        z_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{0});
        switch(init_method)
        {
        case 0: break;
        case 1:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-2, 2});
            break;
        case 2:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<InputDataType>{-0.5, 0.5});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_3<InputDataType>{-0.5, 0.5});
            break;
        case 3:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-5, 5});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            break;
        case 4:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<InputDataType>{2});
            break;
        case 5:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<2>{}); // dy[g0, g1, m, o]
            // dO dot O = [0; 1; 2; ...]
            break;
        case 6:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<3>{}); // dy[g0, g1, m, o]
            // assume mnko = 256
            // P = softmax(QK) = 0.0039 * ones
            // O = P V = 0.0039 * ones
            // dP = dO V = [0, 1, 2, ...; 0, 1, 2, ...; ...]
            // dO dot O = [127.5; ...]
            // dS = P * (dP - dO dot O)
            //
            break;
        default:
            q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
            k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<InputDataType>{});
            ygrad_gs_ms_os.GenerateTensorValue(
                GeneratorTensor_1<InputDataType>{1}); // dy[g0, g1, m, o]
            // assume mnko = 256
            // P = softmax(QK) = 0.0039 * ones
            // O = P V = 0.0039 * ones
            // dP = dO V = ones
            // dS = P * (dP - (dO dot O))
            //    = 0.0039 * ones * (ones - 0.0039*256)
            //    = 0.0039 * ones * (ones - 1)
            //    = 0
        }
        Tensor<InputDataType> q_g_m_k({BatchCount, M, K});
        Tensor<InputDataType> k_g_n_k({BatchCount, N, K});
        Tensor<ZDataType> z_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> v_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> s_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> p_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> y_g_m_o({BatchCount, M, O});
        Tensor<LSEDataType> lse_g_m({BatchCount, M});
        Tensor<InputDataType> p_drop_g_m_n({BatchCount, M, N});

        q_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        k_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            k_g_n_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        v_gs_os_ns.ForEach([&](auto& self, auto idx) {
            v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });

        q_g_m_ks.push_back(q_g_m_k);
        k_g_n_ks.push_back(k_g_n_k);
        z_g_m_ns.push_back(z_g_m_n);
        v_g_n_os.push_back(v_g_n_o);
        s_g_m_ns.push_back(s_g_m_n);
        p_g_m_ns.push_back(p_g_m_n);
        y_g_m_os.push_back(y_g_m_o);
        lse_g_ms.push_back(lse_g_m);
        p_drop_g_m_ns.push_back(p_drop_g_m_n);
        q_tensors.push_back(q_gs_ms_ks);
        k_tensors.push_back(k_gs_ns_ks);
        v_tensors.push_back(v_gs_os_ns);
        y_tensors.push_back(y_gs_ms_os);
        z_tensors.push_back(z_gs_ms_ns);
        lse_tensors.push_back(lse_gs_ms);
        ygrad_tensors.push_back(ygrad_gs_ms_os);
        q_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(InputDataType) * q_gs_ms_ks.GetElementSpaceSize()));
        k_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(InputDataType) * k_gs_ns_ks.GetElementSpaceSize()));
        z_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ZDataType) * z_gs_ms_ns.GetElementSpaceSize()));
        v_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(InputDataType) * v_gs_os_ns.GetElementSpaceSize()));
        y_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(InputDataType) * y_gs_ms_os.GetElementSpaceSize()));
        lse_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(LSEDataType) * lse_gs_ms.GetElementSpaceSize()));
        qgrad_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(OutputDataType) * q_gs_ms_ks.GetElementSpaceSize()));
        kgrad_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(OutputDataType) * k_gs_ns_ks.GetElementSpaceSize()));
        vgrad_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(OutputDataType) * v_gs_os_ns.GetElementSpaceSize()));
        ygrad_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(InputDataType) * y_gs_ms_os.GetElementSpaceSize()));
        q_tensors_device.back()->ToDevice(q_gs_ms_ks.data());
        k_tensors_device.back()->ToDevice(k_gs_ns_ks.data());
        z_tensors_device.back()->ToDevice(z_gs_ms_ns.data());
        v_tensors_device.back()->ToDevice(v_gs_os_ns.data());
        ygrad_tensors_device.back()->ToDevice(ygrad_gs_ms_os.data());
        p_q.push_back(q_tensors_device.back()->GetDeviceBuffer());
        p_k.push_back(k_tensors_device.back()->GetDeviceBuffer());
        p_z.push_back(z_tensors_device.back()->GetDeviceBuffer());
        p_z_nullptr.push_back(nullptr);
        p_v.push_back(v_tensors_device.back()->GetDeviceBuffer());
        p_y.push_back(y_tensors_device.back()->GetDeviceBuffer());
        p_lse.push_back(lse_tensors_device.back()->GetDeviceBuffer());
        p_kgrad.push_back(kgrad_tensors_device.back()->GetDeviceBuffer());
        p_vgrad.push_back(vgrad_tensors_device.back()->GetDeviceBuffer());
        p_ygrad.push_back(ygrad_tensors_device.back()->GetDeviceBuffer());
        p_qgrad.push_back(qgrad_tensors_device.back()->GetDeviceBuffer());
    }
    auto argument =
        gemm.MakeArgument(p_q,
                          p_k,
                          p_z_nullptr,
                          p_v,
                          p_y,
                          p_lse,
                          p_ygrad,
                          p_qgrad,
                          p_kgrad,
                          p_vgrad,
                          {}, // std::array<void*, 1> p_acc0_biases;
                          {}, // std::array<void*, 1> p_acc1_biases;
                          problem_descs,
                          QKVElementOp{},
                          QKVElementOp{},
                          Scale{alpha},
                          QKVElementOp{},
                          YElementOp{},
                          p_drop,
                          std::tuple<unsigned long long, unsigned long long>(seed, offset));

    DeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        // get z matrix
        argument =
            gemm.MakeArgument(p_q,
                              p_k,
                              p_z,
                              p_v,
                              p_y,
                              p_lse,
                              p_ygrad,
                              p_qgrad,
                              p_kgrad,
                              p_vgrad,
                              {}, // std::array<void*, 1> p_acc0_biases;
                              {}, // std::array<void*, 1> p_acc1_biases;
                              problem_descs,
                              QKVElementOp{},
                              QKVElementOp{},
                              Scale{alpha},
                              QKVElementOp{},
                              YElementOp{},
                              p_drop,
                              std::tuple<unsigned long long, unsigned long long>(seed, offset));
        DeviceMem problem_desc_workspace_verify(gemm.GetWorkSpaceSize(&argument));
        gemm.SetWorkSpacePointer(&argument, problem_desc_workspace_verify.GetDeviceBuffer());
        if(!gemm.IsSupportedArgument(argument))
        {
            std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

            return 0;
        }
        invoker.Run(argument, StreamConfig{nullptr, false});

        for(std::size_t i = 0; i < group_count; i++)
        {
            int G1 = v_tensors[i].GetLengths()[1];
            // copy z matirx data form device
            z_tensors_device[i]->FromDevice(z_tensors[i].mData.data());
            z_tensors[i].ForEach([&](auto& self, auto idx) {
                z_g_m_ns[i](idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            run_attention_fwd_host(q_g_m_ks[i],
                                   k_g_n_ks[i],
                                   v_g_n_os[i],
                                   alpha,
                                   s_g_m_ns[i],
                                   p_g_m_ns[i],
                                   y_g_m_os[i],
                                   lse_g_ms[i],
                                   p_drop_g_m_ns[i],
                                   z_g_m_ns[i],
                                   p_dropout_in_16bits,
                                   rp_dropout);

            y_tensors[i].ForEach([&](auto& self, auto idx) {
                self(idx) = y_g_m_os[i](idx[0] * G1 + idx[1], idx[2], idx[3]);
            });
            y_tensors_device[i]->ToDevice(y_tensors[i].data());
            lse_tensors[i].ForEach([&](auto& self, auto idx) {
                self(idx) = lse_g_ms[i](idx[0] * G1 + idx[1], idx[2]);
            });
            lse_tensors_device[i]->ToDevice(lse_tensors[i].data());
            qgrad_tensors_device[i]->SetZero();
            kgrad_tensors_device[i]->SetZero();
            vgrad_tensors_device[i]->SetZero();
        }

        invoker.Run(argument, StreamConfig{nullptr, false});

        for(std::size_t i = 0; i < group_count; i++)
        {

            int G0         = v_tensors[i].GetLengths()[0];
            int G1         = v_tensors[i].GetLengths()[1];
            int O          = v_tensors[i].GetLengths()[2];
            int N          = v_tensors[i].GetLengths()[3];
            int M          = q_tensors[i].GetLengths()[2];
            int K          = q_tensors[i].GetLengths()[3];
            int BatchCount = G0 * G1;
            Tensor<OutputDataType> qgrad_g_m_k({BatchCount, M, K});
            Tensor<OutputDataType> kgrad_g_n_k({BatchCount, N, K});
            Tensor<OutputDataType> vgrad_g_n_o({BatchCount, N, O});
            Tensor<InputDataType> sgrad_g_m_n({BatchCount, M, N});
            Tensor<InputDataType> pgrad_g_m_n({BatchCount, M, N});
            Tensor<InputDataType> pgrad_drop_g_m_n({BatchCount, M, N});
            Tensor<InputDataType> ygrad_g_m_o({BatchCount, M, O});

            ygrad_tensors[i].ForEach([&](auto& self, auto idx) {
                ygrad_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            auto ref_gemm0_grad         = ReferenceGemm0GradInstance{};
            auto ref_gemm0_grad_invoker = ref_gemm0_grad.MakeInvoker();
            using RefGemm0GradArg       = ReferenceGemm0GradInstance::Argument;
            auto ref_gemm1_grad         = ReferenceGemm1GradInstance{};
            auto ref_gemm1_grad_invoker = ref_gemm1_grad.MakeInvoker();
            using RefGemm1GradArg       = ReferenceGemm1GradInstance::Argument;
            // dP = dY * V^T
            auto v_g_o_n = v_g_n_os[i].Transpose({0, 2, 1});
            ref_gemm0_grad_invoker.Run(RefGemm0GradArg{
                ygrad_g_m_o, v_g_o_n, pgrad_drop_g_m_n, PassThrough{}, PassThrough{}, Scale{1.f}});
            auto ref_dropout         = ReferenceDropoutInstance{};
            auto ref_dropout_invoker = ref_dropout.MakeInvoker();
            auto ref_dropout_argment = ref_dropout.MakeArgument(
                z_g_m_ns[i], pgrad_drop_g_m_n, pgrad_g_m_n, p_dropout_in_16bits, rp_dropout);
            ref_dropout_invoker.Run(ref_dropout_argment);

            sgrad_g_m_n.ForEach([&](auto& self, auto idx_gmn) {
                float ygrad_dot_y = 0;
                for(int o = 0; o < O; o++)
                {
                    auto idx_gmo = idx_gmn;
                    idx_gmo[2]   = o;
                    ygrad_dot_y += ck::type_convert<AccDataType>(ygrad_g_m_o(idx_gmo)) *
                                   ck::type_convert<AccDataType>(y_g_m_os[i](idx_gmo));
                }
                self(idx_gmn) = ck::type_convert<InputDataType>(
                    ck::type_convert<AccDataType>(p_g_m_ns[i](idx_gmn)) *
                    (ck::type_convert<AccDataType>(pgrad_g_m_n(idx_gmn)) - ygrad_dot_y));
            });

            auto p_drop_g_n_m = p_drop_g_m_ns[i].Transpose({0, 2, 1});
            ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
                p_drop_g_n_m, ygrad_g_m_o, vgrad_g_n_o, PassThrough{}, PassThrough{}, Scale{1.0f}});
            ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
                sgrad_g_m_n, k_g_n_ks[i], qgrad_g_m_k, PassThrough{}, PassThrough{}, Scale{alpha}});
            auto sgrad_g_n_m = sgrad_g_m_n.Transpose({0, 2, 1});
            ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
                sgrad_g_n_m, q_g_m_ks[i], kgrad_g_n_k, PassThrough{}, PassThrough{}, Scale{alpha}});

            Tensor<OutputDataType> qgrad_gs_ms_ks_host_result(q_tensors[i].GetLengths(),
                                                              q_tensors[i].GetStrides());
            Tensor<OutputDataType> kgrad_gs_ns_ks_host_result(k_tensors[i].GetLengths(),
                                                              k_tensors[i].GetStrides());
            Tensor<OutputDataType> vgrad_gs_os_ns_host_result(v_tensors[i].GetLengths(),
                                                              v_tensors[i].GetStrides());

            Tensor<OutputDataType> qgrad_gs_ms_ks_device_result(q_tensors[i].GetLengths(),
                                                                q_tensors[i].GetStrides());
            Tensor<OutputDataType> kgrad_gs_ns_ks_device_result(k_tensors[i].GetLengths(),
                                                                k_tensors[i].GetStrides());
            Tensor<OutputDataType> vgrad_gs_os_ns_device_result(v_tensors[i].GetLengths(),
                                                                v_tensors[i].GetStrides());

            qgrad_tensors_device[i]->FromDevice(qgrad_gs_ms_ks_device_result.data());
            kgrad_tensors_device[i]->FromDevice(kgrad_gs_ns_ks_device_result.data());
            vgrad_tensors_device[i]->FromDevice(vgrad_gs_os_ns_device_result.data());
            // permute
            qgrad_gs_ms_ks_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = qgrad_g_m_k(g, idx[2], idx[3]);
            });
            kgrad_gs_ns_ks_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = kgrad_g_n_k(g, idx[2], idx[3]);
            });
            vgrad_gs_os_ns_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = vgrad_g_n_o(g, idx[3], idx[2]);
            });

            std::cout << "Checking qgrad:\n";
            pass &= ck::utils::check_err(qgrad_gs_ms_ks_device_result.mData,
                                         qgrad_gs_ms_ks_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
            std::cout << "Checking kgrad:\n";
            pass &= ck::utils::check_err(kgrad_gs_ns_ks_device_result.mData,
                                         kgrad_gs_ns_ks_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
            std::cout << "Checking vgrad:\n";
            pass &= ck::utils::check_err(vgrad_gs_os_ns_device_result.mData,
                                         vgrad_gs_os_ns_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
        }
    }

    return pass ? ((void)(std::cout << "pass\n"), 0) : ((void)(std::cout << "fail\n"), 1);
}

int main(int argc, char* argv[]) { return run(argc, argv); }
