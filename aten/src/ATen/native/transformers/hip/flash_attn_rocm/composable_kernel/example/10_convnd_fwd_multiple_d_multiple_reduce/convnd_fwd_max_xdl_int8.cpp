// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType         = I8;
using BDataType         = I8;
using AccDataType       = I32;
using CShuffleDataType  = I32;
using DsDataType        = ck::Tuple<>;
using EDataType         = I32;
using ReduceAccDataType = I32;
using R0DataType        = I32;
using RsDataType        = ck::Tuple<R0DataType>;

#include "run_convnd_fwd_max_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_fwd_max_example(argc, argv); }
