// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Gemm fused operation. Computes C_m_o = A_m_k * B0_k_n * B1_n_o
                                              |------------|
                                                   Gemm0
                                              |---------------------|
                                                       Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = int8_t;
using B0DataType       = int8_t;
using B1DataType       = int8_t;
using AccDataType      = int32_t;
using CShuffleDataType = int32_t;
using CDataType        = int8_t;

using ALayout  = Row;
using B0Layout = Col;
using B1Layout = Row;
using CLayout  = Row;

using AElementOp    = PassThrough;
using B0ElementOp   = PassThrough;
using Acc0ElementOp = PassThrough;
using B1ElementOp   = PassThrough;
using CElementOp    = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmGemm_Xdl_CShuffle<
    ALayout,
    B0Layout,
    B1Layout,
    CLayout,
    ADataType,
    B0DataType,
    B1DataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    B0ElementOp,
    Acc0ElementOp,
    B1ElementOp,
    CElementOp,
    GemmDefault,
    1,
    256,
    128,         // MPerBlock
    128,         // NPerBlock
    64,          // KPerBlock
    128,         // Gemm1NPerBlock
    64,          // Gemm1KPerBlock
    16,          // AK1
    16,          // BK1
    4,           // B1K1
    32,          // MPerXDL
    32,          // NPerXDL
    1,           // MXdlPerWave
    4,           // NXdlPerWave
    4,           // Gemm1NXdlPerWave
    S<4, 64, 1>, // ABlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    16,
    16,
    true,
    S<4, 64, 1>, // BBlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    16,
    16,
    true,
    S<8, 32, 1>, // B1BlockTransfer
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    4,
    4,
    false,
    1,              // CShuffleMXdlPerWavePerShuffle
    2,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                ADataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                CElementOp>;

using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

#include "run_batched_gemm_gemm_example.inc"

int main(int argc, char* argv[]) { return run_batched_gemm_gemm_example(argc, argv) ? 0 : 1; }
