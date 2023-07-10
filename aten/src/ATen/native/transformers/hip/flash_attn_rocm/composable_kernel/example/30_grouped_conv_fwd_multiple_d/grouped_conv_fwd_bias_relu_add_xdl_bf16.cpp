// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

// kernel data types
using InKernelDataType       = BF16;
using WeiKernelDataType      = BF16;
using AccDataType            = FP32;
using CShuffleDataType       = FP32;
using BiasKernelDataType     = BF16;
using ResidualKernelDataType = BF16;
using OutKernelDataType      = BF16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

#include "run_grouped_conv_fwd_bias_relu_add_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_bias_relu_add_example(argc, argv); }
