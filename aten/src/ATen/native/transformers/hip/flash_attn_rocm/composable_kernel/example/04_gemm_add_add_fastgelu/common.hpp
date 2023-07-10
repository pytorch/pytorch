// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough    = ck::tensor_operation::element_wise::PassThrough;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using I4 = ck::int4_t;
#endif
using I8  = int8_t;
using I32 = int32_t;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA  = 4096;
    ck::index_t StrideB  = 4096;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = 4096;
    ck::index_t StrideE  = 4096;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

inline bool
parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 12)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA  = std::stoi(argv[7]);
        problem_size.StrideB  = std::stoi(argv[8]);
        problem_size.StrideD0 = std::stoi(argv[9]);
        problem_size.StrideD1 = std::stoi(argv[10]);
        problem_size.StrideE  = std::stoi(argv[11]);
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, "
                     "StrideE"
                  << std::endl;
        return false;
    }

    return true;
}
