// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TRANSFORM_BACKWARD_DATA_CONVOLUTION_INTO_GEMM_V4R1_NHWC_KYXC_NHWK_HPP
#define CK_TRANSFORM_BACKWARD_DATA_CONVOLUTION_INTO_GEMM_V4R1_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// Number of GEMMs = YTilde * XTilde
// GemmM = C
// GemmN = N * HTildeSlice * WTildeSlice
// GemmK = K * YDotSlice * XDotSlice
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t IYTildeValue,
          index_t IXTildeValue,
          index_t GemmK1Value>
__host__ __device__ constexpr auto
transform_backward_data_convolution_into_gemm_v4r1_nhwc_kyxc_nhwk(
    const TensorDescriptor<Wei...>& wei_k_y_x_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_ho_wo_k_grid_desc,
    const TensorDescriptor<In...>& in_n_hi_wi_c_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<IYTildeValue>,
    Number<IXTildeValue>,
    Number<GemmK1Value>)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto GemmK1  = Number<GemmK1Value>{};
    constexpr auto IYTilde = Number<IYTildeValue>{};
    constexpr auto IXTilde = Number<IXTildeValue>{};

    const auto N = in_n_hi_wi_c_grid_desc.GetLength(I0);
    const auto C = in_n_hi_wi_c_grid_desc.GetLength(I3);
    const auto K = out_n_ho_wo_k_grid_desc.GetLength(I3);

    const auto Hi = in_n_hi_wi_c_grid_desc.GetLength(I1);
    const auto Wi = in_n_hi_wi_c_grid_desc.GetLength(I2);

    const auto Ho = out_n_ho_wo_k_grid_desc.GetLength(I1);
    const auto Wo = out_n_ho_wo_k_grid_desc.GetLength(I2);

    const auto Y = wei_k_y_x_c_grid_desc.GetLength(I1);
    const auto X = wei_k_y_x_c_grid_desc.GetLength(I2);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
    const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

    const auto YTilde = ConvStrideH / GcdStrideDilationH;
    const auto XTilde = ConvStrideW / GcdStrideDilationW;

    const auto YDot = math::integer_divide_ceil(Y, YTilde);
    const auto XDot = math::integer_divide_ceil(X, XTilde);

    const auto HTilde = Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
    const auto WTilde = Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

    // only work on HTilde and WTilde that contribute to non-padding area of input tensor
    const auto IHTildeSliceBegin = math::integer_divide_floor(
        math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
    const auto IWTildeSliceBegin = math::integer_divide_floor(
        math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

    const auto IHTildeSliceEnd =
        math::min(HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
    const auto IWTildeSliceEnd =
        math::min(WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

    const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
    const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

    // GemmK is different for each GEMM
    const auto YDotSlice = math::integer_divide_ceil(Y - IYTilde, YTilde);
    const auto XDotSlice = math::integer_divide_ceil(X - IXTilde, XTilde);

    const auto K1 = GemmK1;
    const auto K0 = K / K1;

    // weight tensor
    const auto wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
        wei_k_y_x_c_grid_desc,
        make_tuple(make_pass_through_transform(K),
                   make_embed_transform(make_tuple(YDot, YTilde),
                                        make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                   make_embed_transform(make_tuple(XDot, XTilde),
                                        make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto wei_k0_k1_ydotslice_xdotslice_c_grid_desc =
        transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                               make_slice_transform(YDot, I0, YDotSlice),
                                               make_slice_transform(XDot, I0, XDotSlice),
                                               make_freeze_transform(IYTilde),
                                               make_freeze_transform(IXTilde),
                                               make_pass_through_transform(C)),
                                    make_tuple(Sequence<0>{},
                                               Sequence<1>{},
                                               Sequence<3>{},
                                               Sequence<2>{},
                                               Sequence<4>{},
                                               Sequence<5>{}),
                                    make_tuple(Sequence<0, 1>{},
                                               Sequence<2>{},
                                               Sequence<3>{},
                                               Sequence<>{},
                                               Sequence<>{},
                                               Sequence<4>{}));

#if 1
    const auto wei_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
        wei_k0_k1_ydotslice_xdotslice_c_grid_desc,
        make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                   make_pass_through_transform(C),
                   make_pass_through_transform(K1)),
        make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
#else
    const auto wei_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
        wei_k0_k1_ydotslice_xdotslice_c_grid_desc,
        make_tuple(make_merge_transform(make_tuple(K0, YDotSlice, XDotSlice)),
                   make_pass_through_transform(C),
                   make_pass_through_transform(K1)),
        make_tuple(Sequence<0, 2, 3>{}, Sequence<4>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
#endif

    // output tensor
    // this add padding check
    const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
        out_n_ho_wo_k_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Ho, I0, I0),
                   make_pad_transform(Wo, I0, I0),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
        out_n_hop_wop_k_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(YDot, HTilde),
                                        make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                   make_embed_transform(make_tuple(XDot, WTilde),
                                        make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc =
        transform_tensor_descriptor(
            out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_slice_transform(YDot, I0, YDotSlice),
                       make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                       make_slice_transform(XDot, I0, XDotSlice),
                       make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                       make_unmerge_transform(make_tuple(K0, K1))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6>{}));

#if 1
    const auto out_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
        out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc,
        make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                   make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                   make_pass_through_transform(K1)),
        make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
#else
    const auto out_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
        out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc,
        make_tuple(make_merge_transform(make_tuple(K0, YDotSlice, XDotSlice)),
                   make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                   make_pass_through_transform(K1)),
        make_tuple(Sequence<5, 1, 3>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
#endif

    // input tensor
    const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
        in_n_hi_wi_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc = transform_tensor_descriptor(
        in_n_hip_wip_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(YTilde, HTilde),
                                        make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(XTilde, WTilde),
                                        make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto in_n_htildeslice_wtildeslice_c_grid_desc = transform_tensor_descriptor(
        in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_freeze_transform(IYTilde),
                   make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                   make_freeze_transform(IXTilde),
                   make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{},
                   Sequence<1>{},
                   Sequence<2>{},
                   Sequence<3>{},
                   Sequence<4>{},
                   Sequence<5>{}),
        make_tuple(Sequence<0>{},
                   Sequence<>{},
                   Sequence<1>{},
                   Sequence<>{},
                   Sequence<2>{},
                   Sequence<3>{}));

    const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        in_n_htildeslice_wtildeslice_c_grid_desc,
        make_tuple(make_pass_through_transform(C),
                   make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice))),
        make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(wei_gemmk0_gemmm_gemmk1_grid_desc,
                      out_gemmk0_gemmn_gemmk1_grid_desc,
                      in_gemmm_gemmn_grid_desc);
}

} // namespace ck
#endif
