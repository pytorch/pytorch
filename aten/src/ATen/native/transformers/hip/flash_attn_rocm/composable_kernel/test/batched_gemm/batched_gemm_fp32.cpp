// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "profiler/profile_batched_gemm_impl.hpp"

namespace {
using ADataType = float;
using BDataType = float;
using CDataType = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
} // namespace

int main()
{
    int M          = 256;
    int N          = 256;
    int K          = 128;
    int BatchCount = 3;

    bool pass = true;

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Row, Row, Row>(
               true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Row, Col, Row>(
               true, 1, false, 1, M, N, K, K, K, N, M * K, K * N, M * N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Col, Row, Row>(
               true, 1, false, 1, M, N, K, M, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Col, Col, Row>(
               true, 1, false, 1, M, N, K, M, K, N, M * K, K * N, M * N, BatchCount);

    std::cout << "test BatchedGEMM fp32: " << (pass ? "Pass" : "Fail") << std::endl;
    return pass ? 0 : 1;
}
