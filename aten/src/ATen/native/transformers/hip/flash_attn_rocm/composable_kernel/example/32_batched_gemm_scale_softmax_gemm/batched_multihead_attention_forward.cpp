// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
                                                                  |-----------------|
                                                                          Gemm0
                                                          |-------------------------------------|
                                                                          Gemm1
*/

#define DIM 64 // DIM should be a multiple of 8.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
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

using DataType         = BF16;
using GemmDataType     = BF16;
using ADataType        = DataType;
using B0DataType       = DataType;
using B1DataType       = DataType;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = DataType;
using ZDataType        = U16; // INT32
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
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

static constexpr auto TensorSpecA   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB0  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB1  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecC   = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr bool Deterministic = true;

#if(DIM <= 32)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
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
#elif(DIM <= 64)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
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
#elif(DIM <= 128)
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle<
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
#endif

// Ref Gemm0: DataType in, AccDataType out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

// Ref Softmax: AccDataType in, DataType out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

// Ref Gemm1: DataType in, DataType out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

// Ref dropout
using ReferenceDropoutInstance =
    ck::tensor_operation::host::ReferenceDropout<ZDataType, ADataType, ADataType>;

#include "run_batched_multihead_attention_forward.inc"

int main(int argc, char* argv[]) { return run(argc, argv); }
