// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_batched_gemm_add_relu_gemm_add_impl.hpp"
#include "profiler_operation_registry.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

#define OP_NAME "batched_gemm_add_relu_gemm_add"
#define OP_DESC "Batched GEMM+Add+Relu+GEMM+Add"

int profile_batched_gemm_add_relu_gemm_add(int argc, char* argv[])
{
    enum struct GemmMatrixLayout
    {
        MK_NK_MN_NO_MO_MO, // 0
        MK_NK_MN_ON_MO_MO, // 1
    };

    enum struct GemmDataType
    {
        F32_F32_F32_F32_F32_F32, // 0
        F16_F16_F16_F16_F16_F16, // 1
    };

    GemmDataType data_type  = GemmDataType::F16_F16_F16_F16_F16_F16;
    GemmMatrixLayout layout = GemmMatrixLayout::MK_NK_MN_NO_MO_MO;
    bool do_verification    = true;
    int init_method         = 1;
    bool do_log             = 0;
    bool time_kernel        = false;

    // GEMM shape
    ck::index_t M             = 1024;
    ck::index_t N             = 1024;
    ck::index_t K             = 64;
    ck::index_t O             = 128;
    ck::index_t BatchCount    = 4;
    ck::index_t StrideA0      = -1;
    ck::index_t StrideB0      = -1;
    ck::index_t StrideD0      = -1;
    ck::index_t StrideB1      = -1;
    ck::index_t StrideD1      = -1;
    ck::index_t StrideE1      = -1;
    ck::index_t BatchStrideA0 = -1;
    ck::index_t BatchStrideB0 = -1;
    ck::index_t BatchStrideD0 = -1;
    ck::index_t BatchStrideB1 = -1;
    ck::index_t BatchStrideD1 = -1;
    ck::index_t BatchStrideE1 = -1;

    if(argc == 8)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);
    }
    else if(argc == 13)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);

        M          = std::stoi(argv[8]);
        N          = std::stoi(argv[9]);
        K          = std::stoi(argv[10]);
        O          = std::stoi(argv[11]);
        BatchCount = std::stoi(argv[12]);
    }
    else if(argc == 25)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);

        M          = std::stoi(argv[8]);
        N          = std::stoi(argv[9]);
        K          = std::stoi(argv[10]);
        O          = std::stoi(argv[11]);
        BatchCount = std::stoi(argv[12]);

        StrideA0 = std::stoi(argv[13]);
        StrideB0 = std::stoi(argv[14]);
        StrideD0 = std::stoi(argv[15]);
        StrideB1 = std::stoi(argv[16]);
        StrideD1 = std::stoi(argv[17]);
        StrideE1 = std::stoi(argv[18]);

        BatchStrideA0 = std::stoi(argv[19]);
        BatchStrideB0 = std::stoi(argv[20]);
        BatchStrideD0 = std::stoi(argv[21]);
        BatchStrideB1 = std::stoi(argv[22]);
        BatchStrideD1 = std::stoi(argv[23]);
        BatchStrideE1 = std::stoi(argv[24]);
    }
    else
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (1: fp16)\n");
        printf("arg3: matrix layout (0: Relu(A0[m, k] * B0[n, k] + D0[m, n]) * B1[n, o] + D1[m, o] "
               "= E1[m, o]; 1: Relu(A0[m, k] * B0[n, k] + D0[m, n]) * B1[o, n] + D1[m, o] = "
               "E1[m, o];)\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 12: M, N, K, O, Batch\n");
        printf("arg13 to 18: StrideA0, StrideB0, StrideD0, StrideB1, StrideD1, StrideE1\n");
        printf("arg19 to 24: BatchStrideA0, BatchStrideB0, BatchStrideD0, BatchStrideB1, "
               "BatchStrideD1, BatchStrideE1 \n");
        exit(1);
    }

    if(data_type == GemmDataType::F16_F16_F16_F16_F16_F16 &&
       layout == GemmMatrixLayout::MK_NK_MN_NO_MO_MO)
    {
        ck::profiler::profile_batched_gemm_add_relu_gemm_add_impl<Row,            // A0Layout,
                                                                  Col,            // B0Layout,
                                                                  ck::Tuple<Row>, // D0sLayout,
                                                                  Row,            // B1Layout,
                                                                  ck::Tuple<Row>, // D1sLayout,
                                                                  Row,            // E1Layout,
                                                                  F16,            // A0DataType,
                                                                  F16,            // B0DataType,
                                                                  ck::Tuple<F16>, // D0DataType,
                                                                  F16,            // B1DataType,
                                                                  ck::Tuple<F16>, // D1sDataType
                                                                  F16>            // E1DataType,
            (do_verification,
             init_method,
             do_log,
             time_kernel,
             M,
             N,
             K,
             O,
             BatchCount,
             StrideA0,
             StrideB0,
             StrideD0,
             StrideB1,
             StrideD1,
             StrideE1,
             BatchStrideA0,
             BatchStrideB0,
             BatchStrideD0,
             BatchStrideB1,
             BatchStrideD1,
             BatchStrideE1);
    }
    else if(data_type == GemmDataType::F16_F16_F16_F16_F16_F16 &&
            layout == GemmMatrixLayout::MK_NK_MN_ON_MO_MO)
    {
        ck::profiler::profile_batched_gemm_add_relu_gemm_add_impl<Row,            // A0Layout,
                                                                  Col,            // B0Layout,
                                                                  ck::Tuple<Row>, // D0sLayout,
                                                                  Col,            // B1Layout,
                                                                  ck::Tuple<Row>, // D1sLayout,
                                                                  Row,            // E1Layout,
                                                                  F16,            // A0DataType,
                                                                  F16,            // B0DataType,
                                                                  ck::Tuple<F16>, // D0DataType,
                                                                  F16,            // B1DataType,
                                                                  ck::Tuple<F16>, // D1sDataType
                                                                  F16>            // E1DataType,
            (do_verification,
             init_method,
             do_log,
             time_kernel,
             M,
             N,
             K,
             O,
             BatchCount,
             StrideA0,
             StrideB0,
             StrideD0,
             StrideB1,
             StrideD1,
             StrideE1,
             BatchStrideA0,
             BatchStrideB0,
             BatchStrideD0,
             BatchStrideB1,
             BatchStrideD1,
             BatchStrideE1);
    }
    else
    {
        throw std::runtime_error("wrong! this data_type & layout is not implemented");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_add_relu_gemm_add);
