// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_reduce_xdl_common.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

// DataType
using ADataType         = F16;
using BDataType         = F16;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;
using DsDataType        = ck::Tuple<>;
using EDataType         = F16;
using ReduceAccDataType = F32;
using R0DataType        = F32;
using R1DataType        = F32;
using RsDataType        = ck::Tuple<R0DataType, R1DataType>;

// Layout
using ALayout = Row;
using BLayout = Col;
using ELayout = Row;

// Elementwise op
using Square       = ck::tensor_operation::element_wise::UnarySquare;
using Div          = ck::tensor_operation::element_wise::UnaryDivide;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using QsElementOp  = ck::Tuple<PassThrough, Square>;
using RsElementOp  = ck::Tuple<Div, Div>;

// ReduceOp
using R0ThreadReduceOp = ck::reduce::Add;
using R1ThreadReduceOp = ck::reduce::Add;
using RsThreadReduceOp = ck::Tuple<R0ThreadReduceOp, R1ThreadReduceOp>;

static constexpr auto R0GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
static constexpr auto R1GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
using RsGlobalReduceOp = ck::InMemoryDataOperationEnumSequence<R0GlobalReduceOp, R1GlobalReduceOp>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDMultipleR_Xdl_CShuffle
        <ALayout,                   // ALayout
         BLayout,                   // BLayout
         ELayout,                   // ELayout
         ADataType,                 // ADataType
         BDataType,                 // BDataType
         GemmAccDataType,           // GemmAccDataType
         CShuffleDataType,          // CShuffleDataType
         DsDataType,                // DsDataType
         EDataType,                 // EDataType
         ReduceAccDataType,         // ReduceAccDataType
         RsDataType,                // RsDataType
         AElementOp,                // AElementwiseOperation
         BElementOp,                // BElementwiseOperation
         CDEElementOp,              // CDE ElementwiseOperation
         QsElementOp,               // Qs Elementwise Operation
         RsElementOp,               // Rs Elementwise Operation
         RsThreadReduceOp,          // Thread Reduce Operation
         RsGlobalReduceOp,          // Global Reduce Operation
         GemmDefault,               // GEMM Specialization
         1,                         // NumGemmKPrefetchStage
         256,                       // BlockSize
         256,                       // MPerBlock
         128,                       // NPerBlock
         32,                        // KPerBlock
         8,                         // AK1
         8,                         // BK1
         32,                        // MPerXdl
         32,                        // NPerXdl
         4,                         // MXdlPerWave
         2,                         // NXdlPerWave
         S<4, 64, 1>,               // ABlockTransfer ThreadCluster Lengths_K0_M_K1
         S<1, 0, 2>,                // ABlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // ABlockTransfer SrcAccessOrder
         2,                         // ABlockTransfer SrcVectorDim
         8,                         // ABlockTransfer SrcScalarPerVector
         8,                         // ABlockTransfer DstScalarPerVector_K1
         1,                         // ABlockLdsExtraM
         S<4, 64, 1>,               // BBlockTransfer ThreadCluster Lengths_K0_N_K1
         S<1, 0, 2>,                // BBlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // BBlockTransfer SrcAccessOrder
         2,                         // BBlockTransfer SrcVectorDim
         8,                         // BBlockTransfer SrcScalarPerVector
         8,                         // BBlockTransfer DstScalarPerVector_K1
         1,                         // BBlockLdsExtraN
         1,                         // CShuffleMXdlPerWavePerShuffle
         1,                         // CShuffleNXdlPerWavePerShuffle
         S<64, 4>,                  // CD Reduce Thread Transfer ClusterLengths _MPerBlock_NPerBlock
         4,                         // CDE ReduceThreadTransfer ScalarPerVector _NPerBlock
         1>;                        // RThread DstScalarPerVector _MPerBlock
// clang-format on
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        ReduceAccDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CDEElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideE = 1024;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
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
        StrideE = std::stoi(argv[9]);
    }
    else
    {
        std::cout << "arg1: verification (0=no, 1=yes)\n"
                  << " arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
                  << " arg3: Measure kernel execution time (1=ON, 0=Off)\n"
                  << " arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideE\n"
                  << std::endl;
        exit(EXIT_SUCCESS);
    }

    return !run_gemm_reduce_mean_meansquare_xdl<ADataType,
                                                BDataType,
                                                EDataType,
                                                R0DataType,
                                                R1DataType,
                                                ALayout,
                                                BLayout,
                                                ELayout,
                                                AElementOp,
                                                BElementOp,
                                                CDEElementOp,
                                                QsElementOp,
                                                RsElementOp,
                                                RsThreadReduceOp,
                                                ReduceAccDataType,
                                                DeviceOpInstance,
                                                ReferenceGemmInstance>(
        M, N, K, StrideA, StrideB, StrideE, do_verification, init_method, time_kernel);
}
