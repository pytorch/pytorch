// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <tuple>
#include <gtest/gtest.h>

#include "profiler/profile_batchnorm_forward_impl.hpp"

using F16  = ck::half_t;
using F32  = float;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F64  = double;

template <typename Tuple>
class TestBatchNormFwdRank4 : public ::testing::Test
{
    private:
    const double epsilon       = std::numeric_limits<float>::epsilon();
    const double averageFactor = 0.1;

    protected:
    using XDataType       = std::tuple_element_t<0, Tuple>;
    using YDataType       = std::tuple_element_t<1, Tuple>;
    using AccDataType     = std::tuple_element_t<2, Tuple>;
    using ScaleDataType   = std::tuple_element_t<3, Tuple>;
    using BiasDataType    = std::tuple_element_t<4, Tuple>;
    using MeanVarDataType = std::tuple_element_t<5, Tuple>;

    std::vector<std::vector<size_t>> list_of_lengths = {
        {128, 16, 3, 1024}, {128, 16, 6, 512}, {1, 1, 1, 1}, {4, 4, 4, 4}, {32, 32, 32, 32}};
    std::vector<int> reduceDims;

    template <int NumReduceDim>
    void Run()
    {
        for(auto& inOutLengths : list_of_lengths)
        {
            bool pass = true;

            EXPECT_FALSE(reduceDims.size() != NumReduceDim);

            pass =
                pass && ck::profiler::profile_batchnorm_forward_impl<XDataType,
                                                                     YDataType,
                                                                     AccDataType,
                                                                     ScaleDataType,
                                                                     BiasDataType,
                                                                     MeanVarDataType,
                                                                     4,
                                                                     NumReduceDim>(true,
                                                                                   3,
                                                                                   false,
                                                                                   false,
                                                                                   inOutLengths,
                                                                                   reduceDims,
                                                                                   true,
                                                                                   true,
                                                                                   epsilon,
                                                                                   averageFactor);

            pass =
                pass && ck::profiler::profile_batchnorm_forward_impl<XDataType,
                                                                     YDataType,
                                                                     AccDataType,
                                                                     ScaleDataType,
                                                                     BiasDataType,
                                                                     MeanVarDataType,
                                                                     4,
                                                                     NumReduceDim>(true,
                                                                                   3,
                                                                                   false,
                                                                                   false,
                                                                                   inOutLengths,
                                                                                   reduceDims,
                                                                                   false,
                                                                                   false,
                                                                                   epsilon,
                                                                                   averageFactor);

            EXPECT_TRUE(pass);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16, F32, F16, F16, F32>,
                                     std::tuple<F32, F32, F32, F32, F32, F32>,
                                     std::tuple<BF16, BF16, F32, BF16, BF16, F32>,
                                     std::tuple<F64, F64, F64, F64, F64, F64>>;

TYPED_TEST_SUITE(TestBatchNormFwdRank4, KernelTypes);

// nhwc
TYPED_TEST(TestBatchNormFwdRank4, nhwc)
{
    this->reduceDims = {0, 1, 2};
    this->template Run<3>();
}

// nchw
TYPED_TEST(TestBatchNormFwdRank4, nchw)
{
    this->reduceDims = {0, 2, 3};
    this->template Run<3>();
}
