// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <tuple>
#include <gtest/gtest.h>

#include "profiler/profile_conv_fwd_impl.hpp"

template <typename Tuple>
class TestConvndFwd : public ::testing::Test
{
    protected:
    using DataType = std::tuple_element_t<0, Tuple>;
    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        for(auto& param : conv_params)
        {
            bool pass;
            EXPECT_FALSE(conv_params.empty());
            pass = ck::profiler::profile_conv_fwd_impl<
                NDimSpatial,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::NWC,
                                              ck::tensor_layout::convolution::NHWC,
                                              ck::tensor_layout::convolution::NDHWC>>,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::KXC,
                                              ck::tensor_layout::convolution::KYXC,
                                              ck::tensor_layout::convolution::KZYXC>>,
                ck::tuple_element_t<NDimSpatial - 1,
                                    ck::Tuple<ck::tensor_layout::convolution::NWK,
                                              ck::tensor_layout::convolution::NHWK,
                                              ck::tensor_layout::convolution::NDHWK>>,
                DataType,
                DataType,
                DataType>(true,  // do_verification
                          1,     // init_method integer value
                          false, // do_log
                          false, // time_kernel
                          param);
            EXPECT_TRUE(pass);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<float>,
                                     std::tuple<ck::half_t>,
                                     std::tuple<ck::bhalf_t>,
                                     std::tuple<std::int8_t>>;
TYPED_TEST_SUITE(TestConvndFwd, KernelTypes);

// 1d
TYPED_TEST(TestConvndFwd, Conv1dFwd)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 1, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 128, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->template Run<1>();
}

// 2d
TYPED_TEST(TestConvndFwd, Conv2dFwd)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 1, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 1, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}
// 3d
TYPED_TEST(TestConvndFwd, Conv3dFwd)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 1, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 128, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->template Run<3>();
}
