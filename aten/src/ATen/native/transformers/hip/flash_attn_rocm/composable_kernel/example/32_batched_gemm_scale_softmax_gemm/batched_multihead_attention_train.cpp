// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
                                                                  |-----------------|
                                                                          Gemm0
                                                          |-------------------------------------|
                                                                          Gemm1

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

#define PRINT_HOST 0
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
#include "ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_forward_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
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

using InputDataType    = F16;
using OutputDataType   = F16;
using GemmDataType     = F16;
using AccDataType      = F32;
using ShuffleDataType  = F32;
using LSEDataType      = F32;
using ZDataType        = INT32; // INT32
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;
// When OutputDataType == F32,      bwd CShuffleBlockTransferScalarPerVector_NPerBlock = 4
// When OutputDataType == F16/BF16, bwd CShuffleBlockTransferScalarPerVector_NPerBlock = 8
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
using DeviceGemmInstanceFWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        InputDataType,
        InputDataType,
        InputDataType,
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
        S<16, 16, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        2,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        1,              // CShuffleNXdlPerWavePerShuffle
        S<1, 64, 1, 4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,    // MaskingSpecialization
        Deterministic>;

using DeviceGemmInstanceBWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1<
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
using DeviceGemmInstanceFWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        InputDataType,
        InputDataType,
        InputDataType,
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
        S<16, 16, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,    // MaskingSpecialization
        Deterministic>;

using DeviceGemmInstanceBWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1<
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

// using DeviceGemmInstanceBWD =
//     ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V2<
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
using DeviceGemmInstanceFWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        InputDataType,
        InputDataType,
        InputDataType,
        InputDataType,
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
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,    // MaskingSpecialization
        Deterministic>;

using DeviceGemmInstanceBWD =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V2<
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
    const auto mask = DeviceGemmInstanceFWD::C0MatrixMask(N);
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
    ck::index_t M  = 512; // 512
    ck::index_t N  = 512; // 512
    ck::index_t K  = DIM;
    ck::index_t O  = DIM;
    ck::index_t G0 = 4; // 54
    ck::index_t G1 = 6; // 16

    bool input_permute  = false;
    bool output_permute = false;

    float p_drop                    = 0.2;
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
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        p_drop = std::stof(argv[10]);

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
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
    float alpha                   = 1.f / std::sqrt(K);

    std::cout << "do_verification: " << do_verification << std::endl;
    std::cout << "init_method: " << init_method << std::endl;
    std::cout << "time_kernel: " << time_kernel << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "O: " << O << std::endl;
    std::cout << "G0: " << G0 << std::endl;
    std::cout << "G1: " << G1 << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "input_permute: " << input_permute << std::endl;
    std::cout << "output_permute: " << output_permute << std::endl;
    std::cout << "p_drop: " << p_drop << std::endl;
    std::cout << "seed: " << seed << std::endl;
    std::cout << "offset: " << offset << std::endl;

    const ck::index_t BatchCount = G0 * G1;

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
    // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward pass
    // Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
    //    = exp(Si) / exp(log(sum(exp() + ...)))
    //    = exp(Si - log(sum(exp() + ...)))
    //               ^^^^^^^^^^^^^^^^^^^^^
    //                       LSE
    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    Tensor<InputDataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<InputDataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<ZDataType> z_fwd_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<ZDataType> z_bwd_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<InputDataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
    Tensor<InputDataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<InputDataType> y_gs_ms_os_device_result(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);
    Tensor<OutputDataType> qgrad_gs_ms_ks_device_result(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<OutputDataType> kgrad_gs_ns_ks_device_result(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<OutputDataType> vgrad_gs_os_ns_device_result(v_gs_os_ns_lengths, v_gs_os_ns_strides);

    std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
    std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
    std::cout << "z_gs_ms_ns: " << z_fwd_gs_ms_ns.mDesc << std::endl;
    std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;
    std::cout << "y_gs_ms_os: " << y_gs_ms_os_device_result.mDesc << std::endl;
    std::cout << "lse_gs_ms_os: " << lse_gs_ms_device_result.mDesc << std::endl;

    z_fwd_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{0});
    z_bwd_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{0});
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
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1});
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
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<InputDataType>{1}); // dy[g0, g1, m, o]
        // assume mnko = 256
        // P = softmax(QK) = 0.0039 * ones
        // O = P V = 0.0039 * ones
        // dP = dO V = ones
        // dS = P * (dP - (dO dot O))
        //    = 0.0039 * ones * (ones - 0.0039*256)
        //    = 0.0039 * ones * (ones - 1)
        //    = 0
    }

    // qkv gradients have the same descriptor as with qkv
    DeviceMem q_device_buf(sizeof(InputDataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem k_device_buf(sizeof(InputDataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem z_fwd_device_buf(sizeof(ZDataType) * z_fwd_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem z_bwd_device_buf(sizeof(ZDataType) * z_bwd_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem v_device_buf(sizeof(InputDataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem y_device_buf(sizeof(InputDataType) *
                           y_gs_ms_os_device_result.mDesc.GetElementSpaceSize());
    DeviceMem lse_device_buf(sizeof(LSEDataType) *
                             lse_gs_ms_device_result.mDesc.GetElementSpaceSize());
    DeviceMem qgrad_device_buf(sizeof(OutputDataType) *
                               qgrad_gs_ms_ks_device_result.mDesc.GetElementSpaceSize());
    DeviceMem kgrad_device_buf(sizeof(OutputDataType) *
                               kgrad_gs_ns_ks_device_result.mDesc.GetElementSpaceSize());
    DeviceMem vgrad_device_buf(sizeof(OutputDataType) *
                               vgrad_gs_os_ns_device_result.mDesc.GetElementSpaceSize());
    DeviceMem ygrad_device_buf(sizeof(InputDataType) * ygrad_gs_ms_os.mDesc.GetElementSpaceSize());

    q_device_buf.ToDevice(q_gs_ms_ks.mData.data());
    k_device_buf.ToDevice(k_gs_ns_ks.mData.data());
    z_fwd_device_buf.ToDevice(z_fwd_gs_ms_ns.mData.data());
    z_bwd_device_buf.ToDevice(z_bwd_gs_ms_ns.mData.data());
    v_device_buf.ToDevice(v_gs_os_ns.mData.data());
    ygrad_device_buf.ToDevice(ygrad_gs_ms_os.mData.data());

    auto gemm_fwd    = DeviceGemmInstanceFWD{};
    auto invoker_fwd = gemm_fwd.MakeInvoker();
    auto gemm_bwd    = DeviceGemmInstanceBWD{};
    auto invoker_bwd = gemm_bwd.MakeInvoker();

    if(time_kernel)
    {
        auto argument_fwd = gemm_fwd.MakeArgument(
            static_cast<InputDataType*>(q_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(k_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(v_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(y_device_buf.GetDeviceBuffer()),
            static_cast<ZDataType*>(nullptr),
            static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
            q_gs_ms_ks_lengths,
            q_gs_ms_ks_strides,
            k_gs_ns_ks_lengths,
            k_gs_ns_ks_strides,
            v_gs_os_ns_lengths,
            v_gs_os_ns_strides,
            y_gs_ms_os_lengths,
            y_gs_ms_os_strides,
            z_gs_ms_ns_lengths,
            z_gs_ms_ns_strides,
            lse_gs_ms_lengths,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
            QKVElementOp{},
            QKVElementOp{},
            Scale{alpha},
            QKVElementOp{},
            YElementOp{},
            p_drop,          // dropout ratio
            {seed, offset}); // dropout random seed and offset, offset should be at least the number
                             // of elements on a thread

        if(!gemm_fwd.IsSupportedArgument(argument_fwd))
        {
            std::cout << gemm_fwd.GetTypeString() << " does not support this problem" << std::endl;

            return 0;
        }

        float ave_time_fwd = invoker_fwd.Run(argument_fwd, StreamConfig{nullptr, true});

        std::size_t flop_fwd = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
        std::size_t num_btype_fwd =
            (sizeof(InputDataType) * M * K + sizeof(InputDataType) * K * N +
             sizeof(InputDataType) * N * O + sizeof(InputDataType) * M * O) *
            BatchCount;

        float tflops_fwd = static_cast<float>(flop_fwd) / 1.E9 / ave_time_fwd;

        float gb_per_sec_fwd = num_btype_fwd / 1.E6 / ave_time_fwd;

        std::cout << "FWD Perf: " << ave_time_fwd << " ms, " << tflops_fwd << " TFlops, "
                  << gb_per_sec_fwd << " GB/s, " << gemm_fwd.GetTypeString() << std::endl;

        // not need output z matrix
        auto argument_bwd = gemm_bwd.MakeArgument(
            static_cast<InputDataType*>(q_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(k_device_buf.GetDeviceBuffer()),
            static_cast<ZDataType*>(nullptr), // set to nullptr
            static_cast<InputDataType*>(v_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(y_device_buf.GetDeviceBuffer()),
            static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
            static_cast<InputDataType*>(ygrad_device_buf.GetDeviceBuffer()),
            static_cast<OutputDataType*>(qgrad_device_buf.GetDeviceBuffer()),
            static_cast<OutputDataType*>(kgrad_device_buf.GetDeviceBuffer()),
            static_cast<OutputDataType*>(vgrad_device_buf.GetDeviceBuffer()),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
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
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
            QKVElementOp{},
            QKVElementOp{},
            Scale{alpha},
            QKVElementOp{},
            YElementOp{},
            p_drop,
            std::tuple<unsigned long long, unsigned long long>(seed, offset));
        kgrad_device_buf.SetZero(); // reset global accum buffer and rerun
        vgrad_device_buf.SetZero();
        float ave_time_bwd = invoker_bwd.Run(argument_bwd, StreamConfig{nullptr, true});

        // 5 GEMM ops in total:
        // S_MNK / dP_MNO Gemm (Gemm0 rcr)
        // dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)
        // 3x MNK + 2x MNO
        std::size_t flop_bwd = (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;
        // Q/K/V/Y, dQ/dK/dV/dY, LSE
        std::size_t num_btype_bwd =
            (sizeof(InputDataType) * M * K + sizeof(InputDataType) * K * N +
             sizeof(InputDataType) * N * O + sizeof(InputDataType) * M * O * size_t(2) +
             sizeof(OutputDataType) * M * K + sizeof(OutputDataType) * K * N +
             sizeof(OutputDataType) * N * O) *
                BatchCount +
            sizeof(LSEDataType) * M * BatchCount;

        float tflops_bwd = static_cast<float>(flop_bwd) / 1.E9 / ave_time_bwd;

        float gb_per_sec_bwd = num_btype_bwd / 1.E6 / ave_time_bwd;

        std::cout << "BWD Perf: " << ave_time_bwd << " ms, " << tflops_bwd << " TFlops, "
                  << gb_per_sec_bwd << " GB/s, " << gemm_bwd.GetTypeString() << std::endl;
    }

    bool pass = true;
    if(do_verification)
    {
        Tensor<InputDataType> y_gs_ms_os_host_result(y_gs_ms_os_lengths, y_gs_ms_os_strides);
        Tensor<LSEDataType> lse_gs_ms_host_result(lse_gs_ms_lengths, lse_gs_ms_strides);
        Tensor<OutputDataType> qgrad_gs_ms_ks_host_result(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
        Tensor<OutputDataType> kgrad_gs_ns_ks_host_result(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
        Tensor<OutputDataType> vgrad_gs_os_ns_host_result(v_gs_os_ns_lengths, v_gs_os_ns_strides);

        Tensor<InputDataType> q_g_m_k({BatchCount, M, K});
        Tensor<InputDataType> k_g_n_k({BatchCount, N, K});
        Tensor<ZDataType> z_fwd_g_m_n({BatchCount, M, N});
        Tensor<ZDataType> z_bwd_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> v_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> s_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> p_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> p_drop_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> y_g_m_o({BatchCount, M, O});
        Tensor<LSEDataType> lse_g_m({BatchCount, M});

        Tensor<OutputDataType> qgrad_g_m_k({BatchCount, M, K});
        Tensor<OutputDataType> kgrad_g_n_k({BatchCount, N, K});
        Tensor<OutputDataType> vgrad_g_n_o({BatchCount, N, O});
        Tensor<InputDataType> sgrad_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> pgrad_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> pgrad_drop_g_m_n({BatchCount, M, N});
        Tensor<InputDataType> ygrad_g_m_o({BatchCount, M, O});
        Tensor<InputDataType> ygrad_dot_y_g_m({BatchCount, M});

        // get kernel output matrixes
        {
            auto argument_fwd = gemm_fwd.MakeArgument(
                static_cast<InputDataType*>(q_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(k_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(v_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(y_device_buf.GetDeviceBuffer()),
                static_cast<ZDataType*>(z_fwd_device_buf.GetDeviceBuffer()),
                static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
                {}, // std::array<void*, 1> p_acc0_biases;
                {}, // std::array<void*, 1> p_acc1_biases;
                q_gs_ms_ks_lengths,
                q_gs_ms_ks_strides,
                k_gs_ns_ks_lengths,
                k_gs_ns_ks_strides,
                v_gs_os_ns_lengths,
                v_gs_os_ns_strides,
                y_gs_ms_os_lengths,
                y_gs_ms_os_strides,
                z_gs_ms_ns_lengths,
                z_gs_ms_ns_strides,
                lse_gs_ms_lengths,
                {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
                QKVElementOp{},
                QKVElementOp{},
                Scale{alpha},
                QKVElementOp{},
                YElementOp{},
                p_drop,          // dropout ratio
                {seed, offset}); // dropout random seed and offset, offset should be at least the
                                 // number of elements on a thread

            if(!gemm_fwd.IsSupportedArgument(argument_fwd))
            {
                std::cout << gemm_fwd.GetTypeString() << " does not support this problem"
                          << std::endl;

                return 0;
            }
            invoker_fwd.Run(argument_fwd, StreamConfig{nullptr, false});

            // copy fwd matirx data form device
            z_fwd_device_buf.FromDevice(z_fwd_gs_ms_ns.mData.data());
            y_device_buf.FromDevice(y_gs_ms_os_device_result.mData.data());
            lse_device_buf.FromDevice(lse_gs_ms_device_result.mData.data());

            // std::cout << "z_fwd_gs_ms_ns ref:\n" << z_fwd_gs_ms_ns;
            std::ofstream fwd_file("./z_fwd_matrix_txt");
            fwd_file << z_fwd_gs_ms_ns << std::endl;

            kgrad_device_buf.SetZero();
            vgrad_device_buf.SetZero();

            auto argument_bwd = gemm_bwd.MakeArgument(
                static_cast<InputDataType*>(q_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(k_device_buf.GetDeviceBuffer()),
                static_cast<ZDataType*>(z_bwd_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(v_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(y_device_buf.GetDeviceBuffer()),
                static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
                static_cast<InputDataType*>(ygrad_device_buf.GetDeviceBuffer()),
                static_cast<OutputDataType*>(qgrad_device_buf.GetDeviceBuffer()),
                static_cast<OutputDataType*>(kgrad_device_buf.GetDeviceBuffer()),
                static_cast<OutputDataType*>(vgrad_device_buf.GetDeviceBuffer()),
                {}, // std::array<void*, 1> p_acc0_biases;
                {}, // std::array<void*, 1> p_acc1_biases;
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
                {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
                {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
                QKVElementOp{},
                QKVElementOp{},
                Scale{alpha},
                QKVElementOp{},
                YElementOp{},
                p_drop,
                std::tuple<unsigned long long, unsigned long long>(seed, offset));

            if(!gemm_bwd.IsSupportedArgument(argument_bwd))
            {
                std::cout << gemm_bwd.GetTypeString() << " does not support this problem"
                          << std::endl;

                return 0;
            }
            invoker_bwd.Run(argument_bwd, StreamConfig{nullptr, false});

            // copy bwd matirx data form device
            z_bwd_device_buf.FromDevice(z_bwd_gs_ms_ns.mData.data());
            qgrad_device_buf.FromDevice(qgrad_gs_ms_ks_device_result.mData.data());
            kgrad_device_buf.FromDevice(kgrad_gs_ns_ks_device_result.mData.data());
            vgrad_device_buf.FromDevice(vgrad_gs_os_ns_device_result.mData.data());

            // std::cout << "z_bwd_gs_ms_ns ref:\n" << z_bwd_gs_ms_ns;
            std::ofstream bwd_file("./z_bwd_matrix_txt");
            bwd_file << z_bwd_gs_ms_ns << std::endl;
        }

        q_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        k_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            k_g_n_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        v_gs_os_ns.ForEach([&](auto& self, auto idx) {
            v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        z_fwd_gs_ms_ns.ForEach([&](auto& self, auto idx) {
            z_fwd_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });

        run_attention_fwd_host(q_g_m_k,
                               k_g_n_k,
                               v_g_n_o,
                               alpha,
                               s_g_m_n,
                               p_g_m_n,
                               y_g_m_o,
                               lse_g_m,
                               p_drop_g_m_n,
                               z_fwd_g_m_n,
                               p_dropout_in_16bits,
                               rp_dropout);

        ygrad_gs_ms_os.ForEach([&](auto& self, auto idx) {
            ygrad_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        z_bwd_gs_ms_ns.ForEach([&](auto& self, auto idx) {
            z_bwd_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });

#if PRINT_HOST
        {
            std::cout << "q_g_m_k ref:\n" << q_g_m_k;
            std::cout << "k_g_n_k ref:\n" << k_g_n_k;
            std::cout << "v_g_n_o ref:\n" << v_g_n_o;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
        }
#endif

        // Gradients
        auto ref_gemm0_grad         = ReferenceGemm0GradInstance{};
        auto ref_gemm0_grad_invoker = ref_gemm0_grad.MakeInvoker();
        using RefGemm0GradArg       = ReferenceGemm0GradInstance::Argument;
        auto ref_gemm1_grad         = ReferenceGemm1GradInstance{};
        auto ref_gemm1_grad_invoker = ref_gemm1_grad.MakeInvoker();
        using RefGemm1GradArg       = ReferenceGemm1GradInstance::Argument;

        // dP_dropout = dY * V^T
        auto v_g_o_n = v_g_n_o.Transpose({0, 2, 1});
        ref_gemm0_grad_invoker.Run(RefGemm0GradArg{
            ygrad_g_m_o, v_g_o_n, pgrad_drop_g_m_n, PassThrough{}, PassThrough{}, Scale{1.f}});
#if PRINT_HOST
        {
            std::cout << "===== dP = dY * V^T\n";
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "v_g_o_n ref:\n" << v_g_o_n;
            std::cout << "pgrad_drop_g_m_n ref:\n" << pgrad_drop_g_m_n;
        }
#endif
        // dP = dP_dropout x Z
        auto ref_dropout         = ReferenceDropoutInstance{};
        auto ref_dropout_invoker = ref_dropout.MakeInvoker();
        auto ref_dropout_argment = ref_dropout.MakeArgument(
            z_bwd_g_m_n, pgrad_drop_g_m_n, pgrad_g_m_n, p_dropout_in_16bits, rp_dropout);
        ref_dropout_invoker.Run(ref_dropout_argment);

        // dS_i_j = P_i_j .* (dP_i_j - dY_i dot Y_i)
        sgrad_g_m_n.ForEach([&](auto& self, auto idx_gmn) {
            float ygrad_dot_y = 0;
            for(int o = 0; o < O; o++)
            {
                auto idx_gmo = idx_gmn;
                idx_gmo[2]   = o;
                ygrad_dot_y += ck::type_convert<AccDataType>(ygrad_g_m_o(idx_gmo)) *
                               ck::type_convert<AccDataType>(y_g_m_o(idx_gmo));
            }
            self(idx_gmn) = ck::type_convert<InputDataType>(
                ck::type_convert<AccDataType>(p_g_m_n(idx_gmn)) *
                (ck::type_convert<AccDataType>(pgrad_g_m_n(idx_gmn)) - ygrad_dot_y));
        });
#if PRINT_HOST
        {
            std::cout << "===== dS_i_j = P_i_j .* (dP_i_j - dY_i dot Y_i)\n";
            std::cout << "p_g_m_n ref:\n" << p_g_m_n;
            std::cout << "pgrad_g_m_n ref:\n" << pgrad_g_m_n;
            std::cout << "y_g_m_o ref:\n" << y_g_m_o;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "sgrad_g_m_n ref:\n" << sgrad_g_m_n;
        }
#endif
        // dV = P_drop^T * dY
        auto p_drop_g_n_m = p_drop_g_m_n.Transpose({0, 2, 1});
        ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
            p_drop_g_n_m, ygrad_g_m_o, vgrad_g_n_o, PassThrough{}, PassThrough{}, Scale{1.0f}});
#if PRINT_HOST
        {
            std::cout << "===== dV = P^T * dY\n";
            std::cout << "p_drop_g_n_m ref:\n" << p_drop_g_n_m;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "vgrad_g_n_o ref:\n" << vgrad_g_n_o;
        }
#endif

        // dQ = alpha * dS * K
        ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
            sgrad_g_m_n, k_g_n_k, qgrad_g_m_k, PassThrough{}, PassThrough{}, Scale{alpha}});
#if PRINT_HOST
        {
            std::cout << "===== dQ = alpha * dS * K\n";
            std::cout << "sgrad_g_m_n ref:\n" << sgrad_g_m_n;
            std::cout << "k_g_n_k ref:\n" << k_g_n_k;
            std::cout << "qgrad_g_m_k ref:\n" << qgrad_g_m_k;
        }
#endif

        // dK = alpha * dS^T * Q
        auto sgrad_g_n_m = sgrad_g_m_n.Transpose({0, 2, 1});
        ref_gemm1_grad_invoker.Run(RefGemm1GradArg{
            sgrad_g_n_m, q_g_m_k, kgrad_g_n_k, PassThrough{}, PassThrough{}, Scale{alpha}});
#if PRINT_HOST
        {
            std::cout << "===== dK = alpha * dS^T * Q\n";
            std::cout << "sgrad_g_n_m ref:\n" << sgrad_g_n_m;
            std::cout << "q_g_m_k ref:\n" << q_g_m_k;
            std::cout << "kgrad_g_n_k ref:\n" << kgrad_g_n_k;
        }
#endif

        // permute
        y_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = y_g_m_o(g, idx[2], idx[3]);
        });
        lse_gs_ms_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = lse_g_m(g, idx[2]);
        });
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

        // default absolute error and relative error is 0.001
        double rtol = 1e-3;
        double atol = 1e-3;

        // when BF16 is taken, set absolute error and relative error to 0.01
        if(std::is_same_v<InputDataType, ck::bhalf_t> || std::is_same_v<GemmDataType, ck::bhalf_t>)
        {
            rtol = 1e-2;
            atol = 1e-2;
        }

        std::cout << "Checking z:\n";
        pass &= ck::utils::check_err(z_fwd_gs_ms_ns.mData, z_bwd_gs_ms_ns.mData, 1);

        std::cout << "Checking y:\n";
        pass &= ck::utils::check_err(
            y_gs_ms_os_device_result.mData, y_gs_ms_os_host_result.mData, "error", rtol, atol);

        std::cout << "Checking lse:\n";
        pass &= ck::utils::check_err(
            lse_gs_ms_device_result.mData, lse_gs_ms_host_result.mData, "error", rtol, atol);

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

    return pass ? ((void)(std::cout << "pass\n"), 0) : ((void)(std::cout << "fail\n"), 1);
}

int main(int argc, char* argv[]) { return run(argc, argv); }
