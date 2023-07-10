// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_util.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"

#include "gemm_f16_nn_instance.hpp"
#include "gemm_f16_nt_instance.hpp"
#include "gemm_f16_tn_instance.hpp"
#include "gemm_f16_tt_instance.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using F16         = ck::half_t;
using ADataType   = F16;
using BDataType   = F16;
using AccDataType = float;
using CDataType   = F16;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

using ck::gemm_util::GemmParams;
using ck::tensor_operation::device::BaseOperator;
using ck::tensor_operation::device::DeviceGemm;
using namespace ck::tensor_operation::device::instance;

using DeviceGemmNN =
    DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmNT =
    DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmTN =
    DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmTT =
    DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;

struct LayoutConfig
{
    bool ARowMajor;
    bool BRowMajor;
    bool CRowMajor;
};

int main(int argc, char* argv[])
{
    // Class DeviceGemm is templated by layout and precision types so it is not an option to contain
    // them in a single vector. Instead we use abstract BaseOperator class and dynamic_cast() it
    // upon invocation.
    // And since DeviceGemm does not expose template arg information, an extra book keeping class
    // LayoutConfig is used for determining which type a BaseOperator instance should be cast to.
    using OpFactoryFn = void (*)(std::vector<std::unique_ptr<BaseOperator>>&);

    std::vector<std::tuple<GemmParams, LayoutConfig, OpFactoryFn>> problems = {
        // clang-format off
    // 104 tiles
    {GemmParams{2048, 3328, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_256x256},
    {GemmParams{2048, 1664, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_256x128},
    {GemmParams{1024, 1664, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_128x128},
    {GemmParams{1024,  832, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_128x64},
    {GemmParams{2048, 3328, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_256x256},
    {GemmParams{2048, 1664, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_256x128},
    {GemmParams{1024, 1664, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_128x128},
    {GemmParams{1024,  832, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_128x64},
    {GemmParams{2048, 3328, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_256x256},
    {GemmParams{2048, 1664, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_256x128},
    {GemmParams{1024, 1664, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_128x128},
    {GemmParams{1024,  832, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_128x64},
    {GemmParams{2048, 3328, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_256x256},
    {GemmParams{2048, 1664, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_256x128},
    {GemmParams{1024, 1664, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_128x128},
    {GemmParams{1024,  832, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_128x64},
    // 110 tiles
    {GemmParams{2560, 2816, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_256x256},
    {GemmParams{2560, 1408, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_256x128},
    {GemmParams{1280, 1408, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_128x128},
    {GemmParams{1280,  704, 4096}, LayoutConfig{false, false, true}, add_gemm_f16_nn_128x64},
    {GemmParams{2560, 2816, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_256x256},
    {GemmParams{2560, 1408, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_256x128},
    {GemmParams{1280, 1408, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_128x128},
    {GemmParams{1280,  704, 4096}, LayoutConfig{false, true, true}, add_gemm_f16_nt_128x64},
    {GemmParams{2560, 2816, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_256x256},
    {GemmParams{2560, 1408, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_256x128},
    {GemmParams{1280, 1408, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_128x128},
    {GemmParams{1280,  704, 4096}, LayoutConfig{true, false, true}, add_gemm_f16_tn_128x64},
    {GemmParams{2560, 2816, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_256x256},
    {GemmParams{2560, 1408, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_256x128},
    {GemmParams{1280, 1408, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_128x128},
    {GemmParams{1280,  704, 4096}, LayoutConfig{true, true, true}, add_gemm_f16_tt_128x64},
        // clang-format on
    };

    bool do_verification = true;
    bool time_kernel     = true;

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 3)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: time kernel (0=no, 1=yes)" << std::endl;
        return 0;
    }

    bool pass = true;
    for(auto& p : problems)
    {
        GemmParams& problem_size          = std::get<0>(p);
        const LayoutConfig& layout_config = std::get<1>(p);
        const auto& factory               = std::get<2>(p);
        std::vector<std::unique_ptr<BaseOperator>> ops;
        factory(ops);

        // overwrite strides
        problem_size.StrideA = layout_config.ARowMajor ? problem_size.K : problem_size.M;
        problem_size.StrideB = layout_config.BRowMajor ? problem_size.N : problem_size.K;
        problem_size.StrideC = layout_config.CRowMajor ? problem_size.N : problem_size.M;

        if(!layout_config.ARowMajor && !layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmNN*>(ops[0].get());
            pass &= ck::gemm_util::TestGemm<AccDataType>{}(
                op_ptr, problem_size, do_verification, time_kernel);
        }
        else if(!layout_config.ARowMajor && layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmNT*>(ops[0].get());
            pass &= ck::gemm_util::TestGemm<AccDataType>{}(
                op_ptr, problem_size, do_verification, time_kernel);
        }
        else if(layout_config.ARowMajor && !layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmTN*>(ops[0].get());
            pass &= ck::gemm_util::TestGemm<AccDataType>{}(
                op_ptr, problem_size, do_verification, time_kernel);
        }
        else if(layout_config.ARowMajor && layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmTT*>(ops[0].get());
            pass &= ck::gemm_util::TestGemm<AccDataType>{}(
                op_ptr, problem_size, do_verification, time_kernel);
        }
    }

    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return pass ? 0 : 1;
}
