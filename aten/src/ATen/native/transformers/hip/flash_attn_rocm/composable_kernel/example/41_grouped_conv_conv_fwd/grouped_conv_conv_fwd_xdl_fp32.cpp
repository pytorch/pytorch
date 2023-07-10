// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using In0DataType       = float;
using Wei0DataType      = float;
using Acc0DataType      = float;
using Wei1DataType      = float;
using Acc1DataType      = float;
using C1ShuffleDataType = float;
using Out1DataType      = float;

// This is used for reference code
using Out0DataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using In0ElementOp  = ck::tensor_operation::element_wise::PassThrough;
using Wei0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Wei1ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Out0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Out1ElementOp = ck::tensor_operation::element_wise::UnaryConvert;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceBatchedGemmGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmGemm_Xdl_CShuffle<
        Row,               // ALayout
        Col,               // B0Layout
        Col,               // B1Layout
        Row,               // CLayout
        In0DataType,       // ADataType,
        Wei0DataType,      // B0DataType,
        Wei1DataType,      // B1DataType,
        Out1DataType,      // CDataType,
        Acc0DataType,      // AccDataType,
        C1ShuffleDataType, // CShuffleDataType,
        In0ElementOp,      // AElementOp,
        Wei0ElementOp,     // B0ElementOp,
        Out0ElementOp,     // Acc0ElementOp,
        Wei1ElementOp,     // B1ElementOp,
        Out1ElementOp,     // CElementOp,
        GemmDefault,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        16,          // KPerBlock
        128,         // Gemm1NPerBlock
        16,          // Gemm1KPerBlock
        4,           // AK1
        4,           // BK1
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
        4,
        4,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        4,
        4,
        true,
        S<4, 64, 1>, // B1BlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        2,
        2,
        true,
        1,               // CShuffleMXdlPerWavePerShuffle
        2,               // CShuffleNXdlPerWavePerShuffle
        S<1, 16, 1, 16>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4>;              // CShuffleBlockTransferScalarPerVector_NPerBlock

#include "run_grouped_conv_conv_fwd_example.inc"

int main(int argc, char* argv[]) { return run_grouped_conv_conv_fwd_example(argc, argv) ? 0 : 1; }
