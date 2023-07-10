// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType         = BF16;
using BDataType         = BF16;
using AccDataType       = FP32;
using CShuffleDataType  = FP32;
using DsDataType        = ck::Tuple<>;
using EDataType         = BF16;
using ReduceAccDataType = FP32;
using R0DataType        = FP32;
using RsDataType        = ck::Tuple<R0DataType>;

#include "run_convnd_fwd_max_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_fwd_max_example(argc, argv); }
