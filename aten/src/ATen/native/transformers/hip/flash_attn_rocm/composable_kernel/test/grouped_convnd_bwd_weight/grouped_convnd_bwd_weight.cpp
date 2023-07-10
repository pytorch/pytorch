// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_bwd_weight_impl.hpp"

template <typename Tuple>
class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    using DataType = std::tuple_element_t<0, Tuple>;
    std::vector<ck::utils::conv::ConvParam> conv_params;
    ck::index_t split_k{2};

    template <ck::index_t NDimSpatial>
    void Run()
    {
        for(auto& param : conv_params)
        {
            bool pass;
            EXPECT_FALSE(conv_params.empty());
            pass = ck::profiler::profile_grouped_conv_bwd_weight_impl<
                NDimSpatial,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                              ck::tensor_layout::convolution::GNHWC,
                                              ck::tensor_layout::convolution::GNDHWC>>,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                              ck::tensor_layout::convolution::GKYXC,
                                              ck::tensor_layout::convolution::GKZYXC>>,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                              ck::tensor_layout::convolution::GNHWK,
                                              ck::tensor_layout::convolution::GNDHWK>>,
                DataType,
                DataType,
                DataType>(true,  // do_verification
                          1,     // init_method integer value
                          false, // do_log
                          false, // time_kernel
                          param,
                          split_k);
            EXPECT_TRUE(pass);
        }
    }
};

using KernelTypes =
    ::testing::Types<std::tuple<float>, std::tuple<ck::half_t>, std::tuple<ck::bhalf_t>>;
TYPED_TEST_SUITE(TestGroupedConvndBwdWeight, KernelTypes);

TYPED_TEST(TestGroupedConvndBwdWeight, Test1D)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 4, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 4, 64, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 4, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->template Run<1>();
}

TYPED_TEST(TestGroupedConvndBwdWeight, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 4, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 4, 8, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 4, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdWeight, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 4, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 4, 8, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 4, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->template Run<3>();
}
