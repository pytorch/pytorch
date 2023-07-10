// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_V4R4R4_NHWC_KYXC_NHWK_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_V4R4R4_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// A: in
// B: wei
// C: out
// GemmM = N * Do * Ho * Wo
// GemmN = K
// GemmK = Z * Y * X * C
template <typename... In,
          typename... Wei,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmK1Value>
__host__ __device__ constexpr auto
transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk_pad(
    const TensorDescriptor<In...>& in_grid_desc_n_di_hi_wi_c,
    const TensorDescriptor<Wei...>& wei_k_z_y_x_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_do_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<GemmK1Value>)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    constexpr auto GemmK1 = Number<GemmK1Value>{};

    const auto N = in_grid_desc_n_di_hi_wi_c.GetLength(I0);
    const auto K = out_n_do_ho_wo_k_grid_desc.GetLength(I4);
    const auto C = in_grid_desc_n_di_hi_wi_c.GetLength(I4);

    const auto Di = in_grid_desc_n_di_hi_wi_c.GetLength(I1);
    const auto Hi = in_grid_desc_n_di_hi_wi_c.GetLength(I2);
    const auto Wi = in_grid_desc_n_di_hi_wi_c.GetLength(I3);

    const auto Do = out_n_do_ho_wo_k_grid_desc.GetLength(I1);
    const auto Ho = out_n_do_ho_wo_k_grid_desc.GetLength(I2);
    const auto Wo = out_n_do_ho_wo_k_grid_desc.GetLength(I3);

    const auto Z = wei_k_z_y_x_c_grid_desc.GetLength(I1);
    const auto Y = wei_k_z_y_x_c_grid_desc.GetLength(I2);
    const auto X = wei_k_z_y_x_c_grid_desc.GetLength(I3);

    const auto ConvStrideD = conv_strides[I0];
    const auto ConvStrideH = conv_strides[I1];
    const auto ConvStrideW = conv_strides[I2];

    const auto ConvDilationD = conv_dilations[I0];
    const auto ConvDilationH = conv_dilations[I1];
    const auto ConvDilationW = conv_dilations[I2];

    const auto InLeftPadD = in_left_pads[I0];
    const auto InLeftPadH = in_left_pads[I1];
    const auto InLeftPadW = in_left_pads[I2];

    const auto InRightPadD = in_right_pads[I0];
    const auto InRightPadH = in_right_pads[I1];
    const auto InRightPadW = in_right_pads[I2];

    const auto GemmM  = N * Do * Ho * Wo;
    const auto GemmN  = K;
    const auto GemmK  = Z * Y * X * C;
    const auto GemmK0 = GemmK / GemmK1;

    // A: input tensor
    const auto in_grid_desc_n_dip_hip_wip_c = transform_tensor_descriptor(
        in_grid_desc_n_di_hi_wi_c,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Di, InLeftPadD, InRightPadD),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    const auto in_grid_desc_n_z_do_y_ho_x_wo_c = transform_tensor_descriptor(
        in_grid_desc_n_dip_hip_wip_c,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(
            Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}, Sequence<7>{}));

    const auto in_grid_desc_gemmk_gemmm =
        transform_tensor_descriptor(in_grid_desc_n_z_do_y_ho_x_wo_c,
                                    make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                                               make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                                    make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_grid_desc_gemmk0_gemmm_gemmk1 =
        transform_tensor_descriptor(in_grid_desc_gemmk_gemmm,
                                    make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmM)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // B: weight tensor
    const auto wei_grid_desc_gemmk_gemmn = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(Z * Y * X * C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_grid_desc_gemmk0_gemmn_gemmk1 =
        transform_tensor_descriptor(wei_grid_desc_gemmk_gemmn,
                                    make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmN)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // C: output tensor
    const auto out_grid_desc_gemmm_gemmn = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N * Do * Ho * Wo), make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    // const auto out_grid_desc_gemmm_gemmn = transform_tensor_descriptor(
    //     out_n_do_ho_wo_k_grid_desc,
    //     make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
    //                make_pass_through_transform(K)),
    //     make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<3>{}),
    //     make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(in_grid_desc_gemmk0_gemmm_gemmk1,
                      wei_grid_desc_gemmk0_gemmn_gemmk1,
                      out_grid_desc_gemmm_gemmn);
}

} // namespace ck
#endif
