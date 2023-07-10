// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "profiler/profile_batched_gemm_reduce_impl.hpp"

int main()
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    int M = 512;
    int N = 256;
    int K = 128;

    int BatchCount = 3;

    bool pass = true;

    pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                  ck::half_t,
                                                                  ck::half_t,
                                                                  float,
                                                                  Row,
                                                                  Row,
                                                                  Row>(
                       true, 1, false, false, M, N, K, K, N, N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                  ck::half_t,
                                                                  ck::half_t,
                                                                  float,
                                                                  Row,
                                                                  Col,
                                                                  Row>(
                       true, 1, false, false, M, N, K, K, K, N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                  ck::half_t,
                                                                  ck::half_t,
                                                                  float,
                                                                  Col,
                                                                  Row,
                                                                  Row>(
                       true, 1, false, false, M, N, K, M, N, N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                  ck::half_t,
                                                                  ck::half_t,
                                                                  float,
                                                                  Col,
                                                                  Col,
                                                                  Row>(
                       true, 1, false, false, M, N, K, M, K, N, BatchCount);

    if(pass)
    {
        std::cout << "test BatchedGEMM+Reduce fp16: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test BatchedGEMM+Reduce fp16: Fail" << std::endl;
        return -1;
    }
}
