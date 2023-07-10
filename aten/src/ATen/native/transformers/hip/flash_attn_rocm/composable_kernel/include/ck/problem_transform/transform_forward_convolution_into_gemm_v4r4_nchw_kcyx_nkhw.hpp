// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_GEMM_V4R4_NCHW_KCYX_NKHW_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_GEMM_V4R4_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(
    const TensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
    const TensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
    const TensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const auto Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const auto Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const auto Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    // weight tensor
    const auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, C * Y * X)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // input tensor
    const auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
        in_n_c_hi_wi_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
        in_n_c_hip_wip_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc =
        transform_tensor_descriptor(in_n_c_y_ho_x_wo_global_desc,
                                    make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                                               make_merge_transform(make_tuple(N, Ho, Wo))),
                                    make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(
        wei_gemmk_gemmm_global_desc, in_gemmk_gemmn_global_desc, out_gemmm_gemmn_global_desc);
}

template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_no_pad(
    const TensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
    const TensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
    const TensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const auto Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    assert(InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 && InRightPadW == 0);

    // weight tensor
    const auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, C * Y * X)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // input tensor
    const auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
        in_n_c_hi_wi_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc =
        transform_tensor_descriptor(in_n_c_y_ho_x_wo_global_desc,
                                    make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                                               make_merge_transform(make_tuple(N, Ho, Wo))),
                                    make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(
        wei_gemmk_gemmm_global_desc, in_gemmk_gemmn_global_desc, out_gemmm_gemmn_global_desc);
}

template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_1x1(
    const TensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
    const TensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
    const TensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const auto Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    assert(Y == 1 && X == 1 && ConvStrideH == 1 && ConvStrideW == 1 && ConvDilationH == 1 &&
           ConvDilationW == 1 && InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 &&
           InRightPadW == 0);

    // weight tensor
    const auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // input tensor
    const auto in_gemmk_gemmn_global_desc = transform_tensor_descriptor(
        in_n_c_hi_wi_global_desc,
        make_tuple(make_pass_through_transform(C), make_merge_transform(make_tuple(N, Ho, Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2, 3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(
        wei_gemmk_gemmm_global_desc, in_gemmk_gemmn_global_desc, out_gemmm_gemmn_global_desc);
}

} // namespace ck
#endif
