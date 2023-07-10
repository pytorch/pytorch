// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using InDataType = BF16;
// bf16 kernel use fp32 atomic add to accumulate Weight tensor into global memory
using WeiDataType = F32;
using OutDataType = BF16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_bwd_weight_example(argc, argv); }
