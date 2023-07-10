// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/literals.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = ck::int4_t;
using BDataType   = ck::int4_t;
using AccDataType = int32_t;
using CDataType   = int32_t;

using KernelADataType = int8_t;
using KernelBDataType = int8_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdlSplitKCShuffle
    // clang-format off
        <KernelADataType,      //ADataType    
         KernelBDataType,      //BDataType   
         CDataType,            //EDataType
         AccDataType,          //AccDataType
         ALayout,              //ALayout
         BLayout,              //BLayout
         CLayout,              //ELayout
         AElementOp,           //AElementwiseOperation
         BElementOp,           //BElementwiseOperation
         CElementOp,           //CElementwiseOperation
         GemmDefault,          //GEMMSpecialization
         256,                  // BlockSize
         256,                  // MPerBlock
         128,                  // NPerBlock
         4,                    // KPerBlock
         16,                   // K1
         32,                   // MPerXdl
         32,                   // NPerXdl
         4,                    // MXdlPerWave
         2,                    // NXdlPerWave
         S<1, 4, 64, 1>,       // ABlockTransfer ThreadCluster Lengths_K0_M_K1
         S<0, 2, 1, 3>,        // ABlockTransfer ThreadCluster ArrangeOrder
         S<0, 2, 1, 3>,        // ABlockTransfer SrcAccessOrder
         3,                    // ABlockTransfer SrcVectorDim
         16,                   // ABlockTransfer SrcScalarPerVector
         16,                   // ABlockTransfer DstScalarPerVector_K1
         true,                 // ABlockLdsExtraM
         S<1, 4, 64, 1>,       // BBlockTransfer ThreadCluster Lengths_K0_N_K1
         S<0, 1, 3, 2>,        // BBlockTransfer ThreadCluster ArrangeOrder
         S<0, 1, 3, 2>,        // BBlockTransfer SrcAccessOrder
         3,                    // BBlockTransfer SrcVectorDim
         16,                   // BBlockTransfer SrcScalarPerVector
         16,                   // BBlockTransfer DstScalarPerVector_K1
         true,                 // BBlockLdsExtraN
         1,                    // CShuffleMXdlPerWavePerShuffle
         1,                    // CShuffleNXdlPerWavePerShuffle
         S<1, 32, 1, 8>,       // CBlockTransferClusterLengths _MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
         4>;                   // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

#define BUILD_INT4_EXAMPLE
#include "run_splitK_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_splitK_gemm_example(argc, argv); }
