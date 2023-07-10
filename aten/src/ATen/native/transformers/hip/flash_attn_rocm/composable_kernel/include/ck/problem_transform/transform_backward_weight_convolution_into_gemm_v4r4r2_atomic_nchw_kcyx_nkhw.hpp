// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TRANSFORM_BACKWARD_WEIGHT_CONVOLUTION_INTO_GEMM_V4R4R2_ATOMIC_NCHW_KCYX_NKHW_HPP
#define CK_TRANSFORM_BACKWARD_WEIGHT_CONVOLUTION_INTO_GEMM_V4R4R2_ATOMIC_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// GemmM = K
// GemmK = N * Ho * Wo
// GemmN = C * Y * X
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmK1Value,
          typename GemmKBatchType,
          typename GemmKPadType>
__host__ __device__ constexpr auto
transform_backward_weight_convolution_into_gemm_v4r4r2_atomic_nchw_kcyx_nkhw_pad(
    const TensorDescriptor<Wei...>& wei_k_c_y_x_grid_desc,
    const TensorDescriptor<In...>& in_n_c_hi_wi_grid_desc,
    const TensorDescriptor<Out...>& out_n_k_ho_wo_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<GemmK1Value>,
    GemmKBatchType GemmKBatch,
    GemmKPadType GemmKPad)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto GemmK1 = Number<GemmK1Value>{};

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

    const auto GemmM      = K;
    const auto GemmN      = C * Y * X;
    const auto GemmKTotal = N * Ho * Wo;
    const index_t GemmK0  = GemmKPad / (GemmKBatch * GemmK1);

    // A: output tensor
    const auto out_gemmktotal_gemmm_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
        out_gemmktotal_gemmm_grid_desc,
        make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                   make_pass_through_transform(GemmM)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
        out_gemmkpad_gemmm_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1)),
                   make_pass_through_transform(GemmM)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

    // B: input tensor
    const auto in_n_c_hip_wip_grid_desc = transform_tensor_descriptor(
        in_n_c_hi_wi_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n_c_y_ho_x_wo_grid_desc = transform_tensor_descriptor(
        in_n_c_hip_wip_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmktotal_gemmn_grid_desc =
        transform_tensor_descriptor(in_n_c_y_ho_x_wo_grid_desc,
                                    make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                                               make_merge_transform(make_tuple(N, Ho, Wo))),
                                    make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                    make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
        in_gemmktotal_gemmn_grid_desc,
        make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
        in_gemmkpad_gemmn_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

    // C: weight tensor
    const auto wei_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, C * Y * X)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                      in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                      wei_gemmm_gemmn_grid_desc);
}

} // namespace ck
#endif
