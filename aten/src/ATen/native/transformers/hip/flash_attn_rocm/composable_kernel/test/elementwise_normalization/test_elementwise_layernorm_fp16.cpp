// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_elementwise_layernorm_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestElementwiseLayernorm : public ::testing::Test
{
    protected:
    using ADataType     = std::tuple_element_t<0, Tuple>;
    using BDataType     = std::tuple_element_t<1, Tuple>;
    using GammaDataType = std::tuple_element_t<2, Tuple>;
    using BetaDataType  = std::tuple_element_t<3, Tuple>;
    using AccDataType   = std::tuple_element_t<4, Tuple>;
    using YDataType     = std::tuple_element_t<5, Tuple>;

    void Run()
    {
        // M, N
        std::vector<std::vector<ck::index_t>> lengths = {
            {1, 1}, {25, 16}, {39, 777}, {100, 200}, {1024, 1024}, {48 * 256, 2048}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_elementwise_layernorm_impl<ADataType,
                                                                            BDataType,
                                                                            GammaDataType,
                                                                            BetaDataType,
                                                                            AccDataType,
                                                                            YDataType>(
                true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // ADataType, BDataType, GammaDataType, BetaDataType, AccDataType, YDataType>
    std::tuple<F16, F16, F16, F16, F32, F16>>;

TYPED_TEST_SUITE(TestElementwiseLayernorm, KernelTypes);
TYPED_TEST(TestElementwiseLayernorm, Test_FP16) { this->Run(); }
