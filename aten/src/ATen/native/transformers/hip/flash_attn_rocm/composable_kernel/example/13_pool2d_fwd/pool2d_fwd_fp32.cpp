// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "pool2d_fwd_common.hpp"

using InDataType  = float;
using OutDataType = float;
using AccDataType = float;

using IndexDataType = int32_t;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using OutLayout = ck::tensor_layout::convolution::NHWC;

#if 1
static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;
#else
static constexpr auto ReduceOpId = ck::ReduceTensorOp::AVG;
#endif

static constexpr bool OutputIndex  = false;
static constexpr bool PropagateNan = false;

int main(int argc, char* argv[])
{
    bool do_verification;
    int init_method;
    bool time_kernel;

    // Pool shape
    ck::index_t N               = 128;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t window_stride_h = 2;
    ck::index_t window_stride_w = 2;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;

    if(argc == 1)
    {
        do_verification = true;
        init_method     = 1;
        time_kernel     = true;
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = static_cast<bool>(std::stoi(argv[3]));
    }
    else if(argc == 16)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = static_cast<bool>(std::stoi(argv[3]));

        N               = std::stoi(argv[4]);
        C               = std::stoi(argv[5]);
        Y               = std::stoi(argv[6]);
        X               = std::stoi(argv[7]);
        Hi              = std::stoi(argv[8]);
        Wi              = std::stoi(argv[9]);
        window_stride_h = std::stoi(argv[10]);
        window_stride_w = std::stoi(argv[11]);
        in_left_pad_h   = std::stoi(argv[12]);
        in_left_pad_w   = std::stoi(argv[13]);
        in_right_pad_h  = std::stoi(argv[14]);
        in_right_pad_w  = std::stoi(argv[15]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(0);
    }

    bool pass = pool_test<InDataType,
                          OutDataType,
                          AccDataType,
                          IndexDataType,
                          InLayout,
                          OutLayout,
                          ReduceOpId,
                          PropagateNan,
                          OutputIndex>(do_verification,
                                       init_method,
                                       time_kernel,
                                       N,
                                       C,
                                       Y,
                                       X,
                                       Hi,
                                       Wi,
                                       window_stride_h,
                                       window_stride_w,
                                       in_left_pad_h,
                                       in_left_pad_w,
                                       in_right_pad_h,
                                       in_right_pad_w);

    return (pass ? 0 : 1);
}
