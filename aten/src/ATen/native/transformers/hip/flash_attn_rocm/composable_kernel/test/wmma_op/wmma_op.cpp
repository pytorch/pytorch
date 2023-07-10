// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "test/wmma_op/wmma_op_util.hpp"

template <typename SrcType,
          typename DstType,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t AccNum>
bool run_test()
{
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto matmul_default = ck::wmma_op_util::matmul<SrcType, DstType, GPUAccType, AccNum>;
    const auto matmul_swizzle_a =
        ck::wmma_op_util::matmul_swizzle_a<SrcType, DstType, GPUAccType, AccNum>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default, matmul_swizzle_a);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        pass &=
            ck::wmma_op_util::TestWmma<decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
                                       SrcType,
                                       SrcType,
                                       DstType,
                                       GPUAccType,
                                       CPUAccType,
                                       decltype(Row{}),
                                       decltype(Col{}),
                                       decltype(Row{}),
                                       PassThrough,
                                       PassThrough,
                                       PassThrough,
                                       AccNum>{}(std::get<ck::Number<i>{}>(wmma_kernel_container));
    });

    return pass ? 1 : 0;
}
int main(int, char*[])
{
    bool pass = true;
    // clang-format off
    //              |SrcType     |DstType     |GPUAccType  |CPUAccType |AccNum
    pass &= run_test<ck::half_t,  ck::half_t,  float,       float,      8     >();
    pass &= run_test<ck::bhalf_t, ck::bhalf_t, float,       float,      8     >();
    pass &= run_test<ck::half_t,  ck::half_t,  ck::half_t,  ck::half_t, 16    >();
    pass &= run_test<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, float,      16    >();
    pass &= run_test<int8_t,      int8_t,      int32_t,     int32_t,    8     >();
    // clang-format on

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
