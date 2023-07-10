// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_bwd_weight_example(argc, argv); }
