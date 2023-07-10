// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_batched_gemm_reduce_impl.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "batched_gemm_reduce"
#define OP_DESC "Batched GEMM+Reduce"

int profile_batched_gemm_reduce(int argc, char* argv[])
{
    enum struct GemmMatrixLayout
    {
        MK_KN_MN, // 0
        MK_NK_MN, // 1
        KM_KN_MN, // 2
        KM_NK_MN, // 3
    };

    enum struct GemmReduceDataType
    {
        F32_F32_F32_F32_F32, // 0
        F16_F16_F16_F32_F32, // 1
    };

    if(argc != 15)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf("arg8 to 14: M, N, K, StrideA, StrideB, StrideC, BatchCount\n");
        exit(1);
    }

    const auto data_type       = static_cast<GemmReduceDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideC = std::stoi(argv[13]);

    const int BatchCount = std::stoi(argv[14]);

    if(data_type == GemmReduceDataType::F16_F16_F16_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                       ck::half_t,
                                                       ck::half_t,
                                                       float,
                                                       ck::tensor_layout::gemm::RowMajor,
                                                       ck::tensor_layout::gemm::RowMajor,
                                                       ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? K : StrideA,
            (StrideB < 0) ? N : StrideB,
            (StrideC < 0) ? N : StrideC,
            BatchCount);
    }
    else if(data_type == GemmReduceDataType::F16_F16_F16_F32_F32 &&
            layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                       ck::half_t,
                                                       ck::half_t,
                                                       float,
                                                       ck::tensor_layout::gemm::RowMajor,
                                                       ck::tensor_layout::gemm::ColumnMajor,
                                                       ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? K : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            BatchCount);
    }
    else if(data_type == GemmReduceDataType::F16_F16_F16_F32_F32 &&
            layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                       ck::half_t,
                                                       ck::half_t,
                                                       float,
                                                       ck::tensor_layout::gemm::ColumnMajor,
                                                       ck::tensor_layout::gemm::RowMajor,
                                                       ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? N : StrideB,
            (StrideC < 0) ? N : StrideC,
            BatchCount);
    }
    else if(data_type == GemmReduceDataType::F16_F16_F16_F32_F32 &&
            layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                       ck::half_t,
                                                       ck::half_t,
                                                       float,
                                                       ck::tensor_layout::gemm::ColumnMajor,
                                                       ck::tensor_layout::gemm::ColumnMajor,
                                                       ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            BatchCount);
    }
    else
    {
        throw std::runtime_error("wrong! this data_type & layout is not implemented");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_reduce);
