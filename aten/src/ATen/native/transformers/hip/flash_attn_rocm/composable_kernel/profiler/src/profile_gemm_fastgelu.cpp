// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_fastgelu_impl.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "gemm_fastgelu"
#define OP_DESC "GEMM+FastGeLU"

int profile_gemm_fastgelu(int argc, char* argv[])
{
    enum struct MatrixLayout
    {
        MK_KN_MN, // 0
        MK_NK_MN, // 1
        KM_KN_MN, // 2
        KM_NK_MN, // 3
    };

    enum struct MatrixDataType
    {
        F32_F32_F32,    // 0
        F16_F16_F16,    // 1
        BF16_BF16_BF16, // 2
        INT8_INT8_INT8, // 3
    };

    if(argc != 14)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8)\n");
        printf("arg3: matrix layout (0: E[m, n] = FastGeLU(A[m, k] * B[k, n]);\n");
        printf("                     1: E[m, n] = FastGeLU(A[m, k] * B[n, k]);\n");
        printf("                     2: E[m, n] = FastGeLU(A[k, m] * B[k, n]);\n");
        printf("                     3: E[m, n] = FastGeLU(A[k, m] * B[n, k]))\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 13: M, N, K, StrideA, StrideB, StrideE\n");
        // clang-format on
        exit(1);
    }

    const auto data_type       = static_cast<MatrixDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<MatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideE = std::stoi(argv[13]);

    using F16 = ck::half_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto acc_type,
                       auto e_type,
                       auto a_layout,
                       auto b_layout,
                       auto e_layout) {
        using ADataType   = decltype(a_type);
        using BDataType   = decltype(b_type);
        using AccDataType = decltype(acc_type);
        using EDataType   = decltype(e_type);

        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using ELayout = decltype(e_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideE = ck::is_same_v<ELayout, Row> ? N : M;

        bool pass = ck::profiler::profile_gemm_fastgelu_impl<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             ALayout,
                                                             BLayout,
                                                             ELayout>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? DefaultStrideA : StrideA,
            (StrideB < 0) ? DefaultStrideB : StrideB,
            (StrideE < 0) ? DefaultStrideE : StrideE);

        return pass ? 0 : 1;
    };

    if(data_type == MatrixDataType::F16_F16_F16 && layout == MatrixLayout::MK_KN_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == MatrixDataType::F16_F16_F16 && layout == MatrixLayout::MK_NK_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == MatrixDataType::F16_F16_F16 && layout == MatrixLayout::KM_KN_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, Col{}, Row{}, Row{});
    }
    else if(data_type == MatrixDataType::F16_F16_F16 && layout == MatrixLayout::KM_NK_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, Col{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_fastgelu);
