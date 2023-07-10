// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_layernorm_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestLayernorm2d : public ::testing::Test
{
    protected:
    using XDataType     = std::tuple_element_t<0, Tuple>;
    using GammaDataType = std::tuple_element_t<1, Tuple>;
    using BetaDataType  = std::tuple_element_t<2, Tuple>;
    using AccDataType   = std::tuple_element_t<3, Tuple>;
    using YDataType     = std::tuple_element_t<4, Tuple>;

    void Run()
    {
        // [N, D], reduce D
        std::vector<std::vector<ck::index_t>> lengths = {
            {4, 256}, {8, 511}, {9, 1032}, {4, 2048}, {1, 8192}, {4000, 2000}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_layernorm_impl<XDataType,
                                                                GammaDataType,
                                                                BetaDataType,
                                                                AccDataType,
                                                                YDataType,
                                                                2>(true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // XDataType, GammaDataType, BetaDataType, AccDataType, YDataType>
    std::tuple<F16, F16, F16, F32, F16>>;

TYPED_TEST_SUITE(TestLayernorm2d, KernelTypes);
TYPED_TEST(TestLayernorm2d, Test_FP16) { this->Run(); }
