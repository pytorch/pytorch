// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"

namespace {

class TestConvUtil : public ::testing::Test
{
    public:
    void SetNDParams(std::size_t ndims, std::size_t s, std::size_t d, std::size_t p)
    {
        conv_params = ck::utils::conv::ConvParam(ndims,
                                                 2,
                                                 128,
                                                 192,
                                                 256,
                                                 std::vector<ck::index_t>(ndims, 3),
                                                 std::vector<ck::index_t>(ndims, 71),
                                                 std::vector<ck::index_t>(ndims, s),
                                                 std::vector<ck::index_t>(ndims, d),
                                                 std::vector<ck::index_t>(ndims, p),
                                                 std::vector<ck::index_t>(ndims, p));
    }

    protected:
    // -------  default 2D -------
    // input GNCHW {2, 128, 192, 71, 71},
    // weights GKCYX {2, 256, 192, 3, 3},
    // stride {s, s},
    // dilations {d, d},
    // padding {{p, p}, {p, p}
    ck::utils::conv::ConvParam conv_params;
};

} // namespace

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths1D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(1, 2, 1, 1);
    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(1, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71}, "Error: ConvParams 1D stride {1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(1, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37},
                                     "Error: ConvParams 1D padding left/right {2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(1, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D dilation {2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(1, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::index_t>{23},
                             "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths2D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(2, 2, 1, 1);
    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{36, 36},
                                     "Error: ConvParams 2D default constructor."));

    // stride 1, dilation 1, pad 1
    SetNDParams(2, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71, 71}, "Error: ConvParams 2D stride {1,1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(2, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37, 37},
                                     "Error: ConvParams 2D padding left/right {2,2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(2, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36}, "Error: ConvParams 2D dilation {2,2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(2, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::index_t>{23, 23},
                             "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths3D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(3, 2, 1, 1);
    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36, 36}, "Error: ConvParams 3D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(3, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{71, 71, 71},
                                     "Error: ConvParams 3D stride {1, 1, 1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(3, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37, 37, 37},
                                     "Error: ConvParams 3D padding left/right {2, 2, 2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(3, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{36, 36, 36},
                                     "Error: ConvParams 3D dilation {2, 2, 2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(3, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len,
        std::vector<ck::index_t>{23, 23, 23},
        "Error: ConvParams 3D strides{3, 3, 3}, padding {1, 1, 1}, dilations {2, 2, 2}."));
}
