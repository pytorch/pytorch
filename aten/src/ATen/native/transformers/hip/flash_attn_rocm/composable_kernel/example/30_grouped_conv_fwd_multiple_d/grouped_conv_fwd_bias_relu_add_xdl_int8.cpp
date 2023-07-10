// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

// kernel data types
using InKernelDataType       = I8;
using WeiKernelDataType      = I8;
using AccDataType            = I32;
using CShuffleDataType       = I8;
using BiasKernelDataType     = I8;
using ResidualKernelDataType = I8;
using OutKernelDataType      = I8;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

#include "run_grouped_conv_fwd_bias_relu_add_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_bias_relu_add_example(argc, argv); }
