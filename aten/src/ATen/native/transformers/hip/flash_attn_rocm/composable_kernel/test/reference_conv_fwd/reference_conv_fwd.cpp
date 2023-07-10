// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

namespace {

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NDimSpatial,
          typename InDataType    = float,
          typename WeiDataType   = float,
          typename OutDataType   = float,
          typename InLayout      = ck::tensor_layout::convolution::GNHWC,
          typename WeiLayout     = ck::tensor_layout::convolution::GKYXC,
          typename OutLayout     = ck::tensor_layout::convolution::GNHWK,
          typename FillInputOp   = ck::utils::FillMonotonicSeq<InDataType>,
          typename FillWeightsOp = ck::utils::FillConstant<WeiDataType>>
Tensor<OutDataType>
run_reference_convolution_forward(const ck::utils::conv::ConvParam& conv_param,
                                  const FillInputOp& fill_input_op     = FillInputOp{},
                                  const FillWeightsOp& fill_weights_op = FillWeightsOp{0.5f})
{
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weights(wei_g_k_c_xs_desc);
    Tensor<OutDataType> host_output(out_g_n_k_wos_desc);

    fill_input_op(input.begin(), input.end());
    fill_weights_op(weights.begin(), weights.end());
    ck::ranges::fill<OutDataType>(host_output, 0.f);

    auto ref_conv     = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp>();
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(input,
                                              weights,
                                              host_output,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              InElementOp{},
                                              WeiElementOp{},
                                              OutElementOp{});

    ref_invoker.Run(ref_argument);
    return host_output;
}

} // anonymous namespace

// Eeference convolution assume dimensions of tensor descriptors are in GNCDHW/GKCZYX/GNKDHW order,
// regardless of physical tensor layouts in  memory.
// Some tests below assume dimensions of tensor descriptors can be in other order, and therefore
// are disabled
// TODO: add more tests, which comply with assumption about dimension order of reference convolution
// and add tests for more physical layout
#if 0
TEST(ReferenceConvolutionFWD, Conv2DGNHWC)
{
    ck::utils::conv::ConvParam conv_param(2,
                                          1,
                                          1,
                                          1,
                                          2,
                                          std::vector<ck::index_t>{3, 3},
                                          std::vector<ck::index_t>{6, 6},
                                          std::vector<ck::index_t>{1, 1},
                                          std::vector<ck::index_t>{1, 1},
                                          std::vector<ck::index_t>{0, 0},
                                          std::vector<ck::index_t>{0, 0});

    auto out_tensor = run_reference_convolution_forward<2>(conv_param);
    std::vector<std::size_t> ref_dims{1, 1, 4, 4, 1};
    std::vector<float> ref_data{130.5,
                                148.5,
                                166.5,
                                184.5,
                                238.5,
                                256.5,
                                274.5,
                                292.5,
                                346.5,
                                364.5,
                                382.5,
                                400.5,
                                454.5,
                                472.5,
                                490.5,
                                508.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv2DGNHWCStridesDilationsPadding)
{
    ck::utils::conv::ConvParam conv_param(2,
                                          1,
                                          1,
                                          2,
                                          2,
                                          std::vector<ck::index_t>{3, 3},
                                          std::vector<ck::index_t>{12, 12},
                                          std::vector<ck::index_t>{2, 2},
                                          std::vector<ck::index_t>{2, 2},
                                          std::vector<ck::index_t>{1, 1},
                                          std::vector<ck::index_t>{1, 1});

    auto out_tensor                   = run_reference_convolution_forward<2>(conv_param);
    std::vector<std::size_t> ref_dims = std::vector<std::size_t>{1, 5, 5, 2};
    std::vector<float> ref_data{
        210.,  210.,  327.,   327.,   351.,   351.,   375.,   375.,   399.,   399.,
        459.,  459.,  706.5,  706.5,  742.5,  742.5,  778.5,  778.5,  814.5,  814.5,
        747.,  747.,  1138.5, 1138.5, 1174.5, 1174.5, 1210.5, 1210.5, 1246.5, 1246.5,
        1035., 1035., 1570.5, 1570.5, 1606.5, 1606.5, 1642.5, 1642.5, 1678.5, 1678.5,
        1323., 1323., 2002.5, 2002.5, 2038.5, 2038.5, 2074.5, 2074.5, 2110.5, 2110.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DGNWC)
{
    ck::utils::conv::ConvParam conv_param(1,
                                          1,
                                          1,
                                          1,
                                          2,
                                          std::vector<ck::index_t>{3},
                                          std::vector<ck::index_t>{6},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{0},
                                          std::vector<ck::index_t>{0});

    auto out_tensor =
        run_reference_convolution_forward<1,
                                          float,
                                          float,
                                          float,
                                          ck::tensor_layout::convolution::GNWC,
                                          ck::tensor_layout::convolution::GKXC,
                                          ck::tensor_layout::convolution::GNWK>(conv_param);
    std::vector<std::size_t> ref_dims{1, 1, 4, 1};
    std::vector<float> ref_data{7.5, 13.5, 19.5, 25.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DGNWCStridesDilationsPadding)
{
    ck::utils::conv::ConvParam conv_param(1,
                                          1,
                                          1,
                                          2,
                                          2,
                                          std::vector<ck::index_t>{3},
                                          std::vector<ck::index_t>{12},
                                          std::vector<ck::index_t>{2},
                                          std::vector<ck::index_t>{2},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{1});

    auto out_tensor =
        run_reference_convolution_forward<1,
                                          float,
                                          float,
                                          float,
                                          ck::tensor_layout::convolution::GNWC,
                                          ck::tensor_layout::convolution::GKXC,
                                          ck::tensor_layout::convolution::GNWK>(conv_param);
    std::vector<std::size_t> ref_dims{1, 1, 5, 2};
    std::vector<float> ref_data{9., 9., 19.5, 19.5, 31.5, 31.5, 43.5, 43.5, 55.5, 55.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DGNWCSameOutputSize)
{
    ck::utils::conv::ConvParam conv_param(1,
                                          1,
                                          2,
                                          16,
                                          4,
                                          std::vector<ck::index_t>{3},
                                          std::vector<ck::index_t>{16},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{1},
                                          std::vector<ck::index_t>{1});

    auto out_tensor2 = run_reference_convolution_forward<1,
                                                         float,
                                                         float,
                                                         float,
                                                         ck::tensor_layout::convolution::GNWC,
                                                         ck::tensor_layout::convolution::GKXC,
                                                         ck::tensor_layout::convolution::GNWK>(
        conv_param, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});

    std::vector<std::size_t> ref_dims{1, 2, 16, 16};
    std::vector<float> ref_data{
        1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,
        1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,
        3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,
        3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,
        5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,
        5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,
        8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,
        8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,
        10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,
        10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,
        12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001,
        12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001,
        15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,
        15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,
        17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,
        17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,
        20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,
        20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,
        22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,
        22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,
        24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002,
        24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002,
        27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001,
        27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001,
        29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,
        29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,
        32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002,
        32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002,
        34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,
        34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,
        23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,
        23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,
        27.,       27.,       27.,       27.,       27.,       27.,       27.,       27.,
        27.,       27.,       27.,       27.,       27.,       27.,       27.,       27.,
        41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,
        41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,
        44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002,
        44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002,
        46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,
        46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,
        48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998,
        48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998,
        51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,
        51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,
        53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,
        53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,
        56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002,
        56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002,
        58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,
        58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,
        60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998,
        60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998,
        63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,
        63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,
        65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,
        65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,
        68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,
        68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,
        70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,
        70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,
        72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,
        72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,
        49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,
        49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor2.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor2, ref_data, "Error: incorrect results!"));
}
#endif

TEST(ReferenceConvolutionFWD, Conv3DGNCDHW)
{
    ck::utils::conv::ConvParam conv_param(3,
                                          1,
                                          1,
                                          1,
                                          2,
                                          std::vector<ck::index_t>{3, 3, 3},
                                          std::vector<ck::index_t>{6, 6, 6},
                                          std::vector<ck::index_t>{1, 1, 1},
                                          std::vector<ck::index_t>{1, 1, 1},
                                          std::vector<ck::index_t>{0, 0, 0},
                                          std::vector<ck::index_t>{0, 0, 0});

    auto out_tensor = run_reference_convolution_forward<3,
                                                        float,
                                                        float,
                                                        float,
                                                        ck::tensor_layout::convolution::GNCDHW,
                                                        ck::tensor_layout::convolution::GKCZYX,
                                                        ck::tensor_layout::convolution::GNKDHW>(
        conv_param, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});
    std::vector<std::size_t> ref_dims{1, 1, 1, 4, 4, 4};
    std::vector<float> ref_data{
        407.7,     410.40002, 413.09998, 415.80002, 423.90002, 426.6,     429.30002, 432.,
        440.1,     442.80002, 445.5,     448.2,     456.30002, 459.,      461.7,     464.40002,
        504.90002, 507.6,     510.30002, 513.,      521.1,     523.8,     526.5,     529.2001,
        537.3,     540.,      542.7001,  545.4,     553.5,     556.2001,  558.9,     561.6,
        602.10004, 604.8,     607.5,     610.2,     618.3,     621.,      623.7,     626.4,
        634.5,     637.2,     639.9,     642.60004, 650.7,     653.4,     656.10004, 658.8,
        699.3,     702.,      704.7,     707.4,     715.5,     718.2,     720.9,     723.60004,
        731.7,     734.4001,  737.10004, 739.8,     747.9001,  750.60004, 753.3,     756.};
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mDesc.GetLengths(),
                                     ref_dims,
                                     "Error [case 1]: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor, ref_data, "Error [case 1]: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv3DGNCDHWStridesDilations)
{
    ck::utils::conv::ConvParam conv_param(3,
                                          1,
                                          1,
                                          2,
                                          2,
                                          std::vector<ck::index_t>{3, 3, 3},
                                          std::vector<ck::index_t>{12, 12, 12},
                                          std::vector<ck::index_t>{3, 3, 3},
                                          std::vector<ck::index_t>{1, 1, 1},
                                          std::vector<ck::index_t>{0, 0, 0},
                                          std::vector<ck::index_t>{0, 0, 0});

    auto out_tensor = run_reference_convolution_forward<3,
                                                        float,
                                                        float,
                                                        float,
                                                        ck::tensor_layout::convolution::GNCDHW,
                                                        ck::tensor_layout::convolution::GKCZYX,
                                                        ck::tensor_layout::convolution::GNKDHW>(
        conv_param, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});
    std::vector<std::size_t> ref_dims{1, 1, 2, 4, 4, 4};
    std::vector<float> ref_data{
        2756.7002, 2764.7998, 2772.9001, 2781.,     2853.9001, 2862.,     2870.1,    2878.2002,
        2951.1,    2959.2002, 2967.2998, 2975.4001, 3048.2998, 3056.4001, 3064.5,    3072.6,
        3923.1,    3931.2,    3939.2998, 3947.4,    4020.2998, 4028.4001, 4036.5002, 4044.5999,
        4117.5,    4125.6,    4133.7,    4141.8,    4214.7,    4222.8,    4230.9004, 4239.,
        5089.5,    5097.5996, 5105.7,    5113.8,    5186.7,    5194.8,    5202.9,    5211.,
        5283.9004, 5292.,     5300.0996, 5308.2,    5381.0996, 5389.2,    5397.3,    5405.4004,
        6255.9004, 6264.0005, 6272.1,    6280.2,    6353.1,    6361.2,    6369.301,  6377.4,
        6450.301,  6458.4,    6466.5,    6474.6,    6547.5,    6555.6,    6563.699,  6571.801,
        2756.7002, 2764.7998, 2772.9001, 2781.,     2853.9001, 2862.,     2870.1,    2878.2002,
        2951.1,    2959.2002, 2967.2998, 2975.4001, 3048.2998, 3056.4001, 3064.5,    3072.6,
        3923.1,    3931.2,    3939.2998, 3947.4,    4020.2998, 4028.4001, 4036.5002, 4044.5999,
        4117.5,    4125.6,    4133.7,    4141.8,    4214.7,    4222.8,    4230.9004, 4239.,
        5089.5,    5097.5996, 5105.7,    5113.8,    5186.7,    5194.8,    5202.9,    5211.,
        5283.9004, 5292.,     5300.0996, 5308.2,    5381.0996, 5389.2,    5397.3,    5405.4004,
        6255.9004, 6264.0005, 6272.1,    6280.2,    6353.1,    6361.2,    6369.301,  6377.4,
        6450.301,  6458.4,    6466.5,    6474.6,    6547.5,    6555.6,    6563.699,  6571.801};
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mDesc.GetLengths(),
                                     ref_dims,
                                     "Error [case 2]: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor, ref_data, "Error [case 2]: incorrect results!", 1e-4f, 1e-6f));
}
