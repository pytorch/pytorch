// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_CONTRACTION_V6R1_NCHW_KCYX_NKHW_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_CONTRACTION_V6R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// GemmM0 = 1
// GemmM1 = K
// GemmN0 = N0
// GemmN1 = (N / N0) * Ho * Wo
// GemmK0 = (C / C0) * Y * X
// GemmK1 = C0
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          typename N0Type,
          typename C0Type>
__host__ __device__ constexpr auto
transform_forward_convolution_into_contraction_v6r1_nchw_kcyx_nkhw_pad(
    const TensorDescriptor<Wei...>& wei_k_c_y_x_grid_desc,
    const TensorDescriptor<In...>& in_n_c_hi_wi_grid_desc,
    const TensorDescriptor<Out...>& out_n_k_ho_wo_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const N0Type& N0,
    const C0Type& C0)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_grid_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_grid_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_grid_desc.GetLength(I1);

    const auto Hi = in_n_c_hi_wi_grid_desc.GetLength(I2);
    const auto Wi = in_n_c_hi_wi_grid_desc.GetLength(I3);

    const auto Ho = out_n_k_ho_wo_grid_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_grid_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_grid_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_grid_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    const auto N1 = N / N0;
    const auto C1 = C / C0;

    // weight tensor
    const auto wei_gk0_gm0_gm1_gk1_grid_desc =
        transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C * Y * X)),
                                    make_tuple(make_unmerge_transform(make_tuple(I1, K)),
                                               make_unmerge_transform(make_tuple(C0, C1 * Y * X))),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<1, 2>{}, Sequence<3, 0>{}));

    // input tensor
    const auto in_n_c_hip_wip_grid_desc = transform_tensor_descriptor(
        in_n_c_hi_wi_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n0_n1_c0_c1_y_ho_x_wo_grid_desc = transform_tensor_descriptor(
        in_n_c_hip_wip_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(N0, N1)),
                   make_unmerge_transform(make_tuple(C0, C1)),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

    const auto in_gk0_gn0_gn1_gk1_grid_desc = transform_tensor_descriptor(
        in_n0_n1_c0_c1_y_ho_x_wo_grid_desc,
        make_tuple(make_merge_transform(make_tuple(C1, Y, X)),
                   make_pass_through_transform(N0),
                   make_merge_transform(make_tuple(N1, Ho, Wo)),
                   make_pass_through_transform(C0)),
        make_tuple(Sequence<3, 4, 6>{}, Sequence<0>{}, Sequence<1, 5, 7>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    // output tensor
    const auto out_n_k_howo_grid_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho * Wo));

    const auto out_n0_n1_1_k_howo_grid_desc =
        transform_tensor_descriptor(out_n_k_howo_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(N0, N1)),
                                               make_unmerge_transform(make_tuple(I1, K)),
                                               make_pass_through_transform(Ho * Wo)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4>{}));

    const auto out_gm0_gm1_gn0_gn1_grid_desc = transform_tensor_descriptor(
        out_n0_n1_1_k_howo_grid_desc,
        make_tuple(make_pass_through_transform(I1),
                   make_pass_through_transform(K),
                   make_pass_through_transform(N0),
                   make_merge_transform_v2_magic_division(make_tuple(N1, Ho * Wo))),
        make_tuple(Sequence<2>{}, Sequence<3>{}, Sequence<0>{}, Sequence<1, 4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    return make_tuple(
        wei_gk0_gm0_gm1_gk1_grid_desc, in_gk0_gn0_gn1_gk1_grid_desc, out_gm0_gm1_gn0_gn1_grid_desc);
}

} // namespace ck
#endif
