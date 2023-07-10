// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_gemm_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
    MK_KN_NM, // 4
    MK_NK_NM, // 5
    KM_KN_NM, // 6
    KM_NK_NM, // 7
};

enum struct GemmDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

std::vector<int> argToIntArray(char* input)
{
    std::vector<int> out;

    std::istringstream in(input);

    std::string item;

    while(std::getline(in, item, ','))
    {
        out.push_back(std::stoi(item));
    }

    return out;
}

#define OP_NAME "grouped_gemm"
#define OP_DESC "Grouped GEMM"

int profile_grouped_gemm(int argc, char* argv[])
{
    if(!(argc == 14))
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf("arg8 to 13: Ms, Ns, Ks, StrideAs, StrideBs, StrideCs (e.g., 256,256 128,128 64,64 "
               "64,64 64,64 128,128)\n");
        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const auto Ms = argToIntArray(argv[8]);
    const auto Ns = argToIntArray(argv[9]);
    const auto Ks = argToIntArray(argv[10]);

    const auto StrideAs = argToIntArray(argv[11]);
    const auto StrideBs = argToIntArray(argv[12]);
    const auto StrideCs = argToIntArray(argv[13]);

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_impl<ck::half_t,
                                                ck::half_t,
                                                ck::half_t,
                                                float,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                                   init_method,
                                                                                   do_log,
                                                                                   time_kernel,
                                                                                   Ms,
                                                                                   Ns,
                                                                                   Ks,
                                                                                   StrideAs,
                                                                                   StrideBs,
                                                                                   StrideCs);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_impl<ck::half_t,
                                                ck::half_t,
                                                ck::half_t,
                                                float,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::ColumnMajor,
                                                ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                                   init_method,
                                                                                   do_log,
                                                                                   time_kernel,
                                                                                   Ms,
                                                                                   Ns,
                                                                                   Ks,
                                                                                   StrideAs,
                                                                                   StrideBs,
                                                                                   StrideCs);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_impl<ck::half_t,
                                                ck::half_t,
                                                ck::half_t,
                                                float,
                                                ck::tensor_layout::gemm::ColumnMajor,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                                   init_method,
                                                                                   do_log,
                                                                                   time_kernel,
                                                                                   Ms,
                                                                                   Ns,
                                                                                   Ks,
                                                                                   StrideAs,
                                                                                   StrideBs,
                                                                                   StrideCs);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_impl<ck::half_t,
                                                ck::half_t,
                                                ck::half_t,
                                                float,
                                                ck::tensor_layout::gemm::ColumnMajor,
                                                ck::tensor_layout::gemm::ColumnMajor,
                                                ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                                   init_method,
                                                                                   do_log,
                                                                                   time_kernel,
                                                                                   Ms,
                                                                                   Ns,
                                                                                   Ks,
                                                                                   StrideAs,
                                                                                   StrideBs,
                                                                                   StrideCs);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_gemm);
