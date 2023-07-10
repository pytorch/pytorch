// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"

class TestGroupedConvNdFwd : public ::testing::Test
{
    protected:
    std::vector<ck::utils::conv::ConvParam> conv_params;
};

// 1d GNWC/GKXC/GNWK
TEST_F(TestGroupedConvNdFwd, GroupedConv1dFwdGNWC)
{
    conv_params.clear();
    conv_params.push_back({1, 2, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    conv_params.push_back({1, 2, 128, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    conv_params.push_back({1, 2, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_grouped_conv_fwd_impl<1,
                                                           ck::tensor_layout::convolution::GNWC,
                                                           ck::tensor_layout::convolution::GKXC,
                                                           ck::tensor_layout::convolution::GNWK,
                                                           float,
                                                           float,
                                                           float>(true,  // do_verification
                                                                  1,     // init_method
                                                                  false, // do_log
                                                                  false, // time_kernel
                                                                  param);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<1,
                                                           ck::tensor_layout::convolution::GNWC,
                                                           ck::tensor_layout::convolution::GKXC,
                                                           ck::tensor_layout::convolution::GNWK,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<1,
                                                           ck::tensor_layout::convolution::GNWC,
                                                           ck::tensor_layout::convolution::GKXC,
                                                           ck::tensor_layout::convolution::GNWK,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t>(true,  // do_verification
                                                                        1,     // init_method
                                                                        false, // do_log
                                                                        false, // time_kernel
                                                                        param);

        EXPECT_TRUE(pass);

        // int8
        pass = ck::profiler::profile_grouped_conv_fwd_impl<1,
                                                           ck::tensor_layout::convolution::GNWC,
                                                           ck::tensor_layout::convolution::GKXC,
                                                           ck::tensor_layout::convolution::GNWK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t>(true,  // do_verification
                                                                   1,     // init_method
                                                                   false, // do_log
                                                                   false, // time_kernel
                                                                   param);

        EXPECT_TRUE(pass);
    }
}

// 2d GNHWC/GKYXC/GNHWK
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdGNHWC)
{
    conv_params.clear();
    conv_params.push_back({2, 2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    conv_params.push_back({2, 2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::GNHWC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::GNHWK,
                                                           float,
                                                           float,
                                                           float>(true,  // do_verification
                                                                  1,     // init_method
                                                                  false, // do_log
                                                                  false, // time_kernel
                                                                  param);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::GNHWC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::GNHWK,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::GNHWC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::GNHWK,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t>(true,  // do_verification
                                                                        1,     // init_method
                                                                        false, // do_log
                                                                        false, // time_kernel
                                                                        param);

        EXPECT_TRUE(pass);

        // int8
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::GNHWC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::GNHWK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t>(true,  // do_verification
                                                                   1,     // init_method
                                                                   false, // do_log
                                                                   false, // time_kernel
                                                                   param);

        EXPECT_TRUE(pass);
    }
}

// 3d GNDHWC/GKZYXC/GNDHWK
TEST_F(TestGroupedConvNdFwd, GroupedConv3dFwdGNDHWC)
{
    conv_params.clear();
    conv_params.push_back(
        {3, 2, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    conv_params.push_back(
        {3, 2, 128, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    conv_params.push_back(
        {3, 2, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_grouped_conv_fwd_impl<3,
                                                           ck::tensor_layout::convolution::GNDHWC,
                                                           ck::tensor_layout::convolution::GKZYXC,
                                                           ck::tensor_layout::convolution::GNDHWK,
                                                           float,
                                                           float,
                                                           float>(true,  // do_verification
                                                                  1,     // init_method
                                                                  false, // do_log
                                                                  false, // time_kernel
                                                                  param);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<3,
                                                           ck::tensor_layout::convolution::GNDHWC,
                                                           ck::tensor_layout::convolution::GKZYXC,
                                                           ck::tensor_layout::convolution::GNDHWK,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<3,
                                                           ck::tensor_layout::convolution::GNDHWC,
                                                           ck::tensor_layout::convolution::GKZYXC,
                                                           ck::tensor_layout::convolution::GNDHWK,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t>(true,  // do_verification
                                                                        1,     // init_method
                                                                        false, // do_log
                                                                        false, // time_kernel
                                                                        param);

        EXPECT_TRUE(pass);

        // int8
        pass = ck::profiler::profile_grouped_conv_fwd_impl<3,
                                                           ck::tensor_layout::convolution::GNDHWC,
                                                           ck::tensor_layout::convolution::GKZYXC,
                                                           ck::tensor_layout::convolution::GNDHWK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t>(true,  // do_verification
                                                                   1,     // init_method
                                                                   false, // do_log
                                                                   false, // time_kernel
                                                                   param);

        EXPECT_TRUE(pass);
    }
}

// 2d NHWGC/KYXGC/NHWGK
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNHWGC)
{
    conv_params.clear();
    conv_params.push_back({2, 2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    conv_params.push_back({2, 2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NHWGC,
                                                           ck::tensor_layout::convolution::KYXGC,
                                                           ck::tensor_layout::convolution::NHWGK,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param);

        EXPECT_TRUE(pass);
    }
}
