// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "cgemm_xdl_common.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_cgemm.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_cgemm_4gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

using ADataType   = BF16;
using BDataType   = BF16;
using CDataType   = BF16;
using AccDataType = F32;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using ReferenceCGemmInstance = ck::tensor_operation::host::
    ReferenceCGemm<ADataType, BDataType, CDataType, PassThrough, PassThrough, PassThrough>;

// clang-format off
using DeviceCGemmInstance = ck::tensor_operation::device::DeviceCGemm_4Gemm_Xdl_CShuffle
    <ALayout,                    // typename ALayout
     BLayout,                    // typename BLayout
     CLayout,                    // typename CLayout
     ADataType,                  // typename ADataType
     BDataType,                  // typename BDataType
     CDataType,                  // typename CDataType
     AccDataType,                // typename GemmAccDataType
     CDataType,                  // typename CShuffleDataType
     PassThrough,                // typename AElementwiseOperation
     PassThrough,                // typename BElementwiseOperation
     PassThrough,                // typename CElementwiseOperation
     GemmDefault,                // GemmSpecialization GemmSpec
     1,                          // index_t NumGemmKPrefetchStage
     256,                        // index_t BlockSize
     256,                        // index_t MPerBlock
     128,                        // index_t NPerBlock
     32,                         // index_t KPerBlock
     8,                          // index_t AK1
     8,                          // index_t BK1
     32,                         // index_t MPerXDL
     32,                         // index_t NPerXDL
     4,                          // index_t MXdlPerWave
     2,                          // index_t NXdlPerWave
     S<4, 64, 1>,                // typename ABlockTransferThreadClusterLengths_AK0_M_AK1
     S<1, 0, 2>,                 // typename ABlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename ABlockTransferSrcAccessOrder
     2,                          // index_t ABlockTransferSrcVectorDim
     8,                          // index_t ABlockTransferSrcScalarPerVector
     8,                          // index_t ABlockTransferDstScalarPerVector_AK1
     1,                          // index_t ABlockLdsExtraM
     S<4, 64, 1>,                // typename BBlockTransferThreadClusterLengths_BK0_N_BK1
     S<1, 0, 2>,                 // typename BBlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename BBlockTransferSrcAccessOrder
     2,                          // index_t BBlockTransferSrcVectorDim
     8,                          // index_t BBlockTransferSrcScalarPerVector
     8,                          // index_t BBlockTransferDstScalarPerVector_BK1
     1,                          // index_t BBlockLdsExtraN
     1,                          // index_t CShuffleMXdlPerWavePerShuffle
     1,                          // index_t CShuffleNXdlPerWavePerShuffle
     S<1, 32, 1, 8>,             // typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
     8>;                         // index_t CShuffleBlockTransferScalarPerVector_NPerBlock
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // CGEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 416;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        std::cout << "arg1: verification (0=no, 1=yes)\n"
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
                  << "arg3: run kernel # of times (>1)\n"
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n"
                  << std::endl;
        exit(0);
    }

    return !run_cgemm_xdl<ADataType,
                          BDataType,
                          CDataType,
                          ALayout,
                          BLayout,
                          CLayout,
                          PassThrough,
                          PassThrough,
                          PassThrough,
                          DeviceCGemmInstance,
                          ReferenceCGemmInstance>(
        M, N, K, StrideA, StrideB, StrideC, do_verification, init_method, time_kernel);
}
