// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {
namespace tensor_operation {

template <
    index_t NDimSpatial,
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization,
    index_t AK1,
    index_t BK1,
    index_t GemmMPerBlock,
    index_t GemmNPerBlock,
    bool DoPadGemmM,
    bool DoPadGemmN>
struct TransformConvBwdDataToGemm_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          is_same_v<ALayout, tensor_layout::convolution::GNHWK>,
                                      bool>::type = false>
    static auto MakeADescriptor_AK0_M_AK1(
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ytilde = tildes[0];
        index_t i_xtilde = tildes[1];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t K = wei_g_k_c_xs_lengths[1];

        const index_t Hi = in_g_n_c_wis_lengths[3];
        const index_t Wi = in_g_n_c_wis_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t AK0 = K / AK1;

        // assume packed
        const auto out_n_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmak0_gemmmraw_gemmak1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K)),
                make_tuple(make_pass_through_transform(N * Ho * Wo),
                           make_unmerge_transform(make_tuple(AK0, AK1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            const auto out_gemmak0_gemmm_gemmak1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmak0_gemmmraw_gemmak1_grid_desc,
                    make_tuple(AK0, GemmMPerBlock, AK1),
                    Sequence<false, DoPadGemmM, false>{});

            return out_gemmak0_gemmm_gemmak1_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // GemmK is different for each GEMM
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // A: output tensor
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
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(YDot, HTilde),
                                         make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                    make_embed_transform(make_tuple(XDot, WTilde),
                                         make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                    make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_ak0_ak1_grid_desc =
                transform_tensor_descriptor(
                    out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_slice_transform(YDot, I0, YDotSlice),
                               make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                               make_slice_transform(XDot, I0, XDotSlice),
                               make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                               make_unmerge_transform(make_tuple(AK0, AK1))),
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

            const auto out_gemmak0_gemmmraw_gemmak1_grid_desc = transform_tensor_descriptor(
                out_n_ydotslice_htildeslice_xdotslice_wtildeslice_ak0_ak1_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, AK0)),
                           make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(AK1)),
                make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto out_gemmak0_gemmm_gemmak1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmak0_gemmmraw_gemmak1_grid_desc,
                    make_tuple(AK0, GemmMPerBlock, AK1),
                    Sequence<false, DoPadGemmM, false>{});

            return out_gemmak0_gemmm_gemmak1_grid_desc;
        }
    }

    template <typename BLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC>,
                                      bool>::type = false>
    static auto MakeBDescriptor_BK0_N_BK1(
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& /* input_left_pads */,
        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ytilde = tildes[0];
        index_t i_xtilde = tildes[1];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t K = wei_g_k_c_xs_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t BK0 = K / BK1;

        // assume packed
        const auto wei_k_y_x_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            // B: weight tensor
            const auto wei_gemmbk0_gemmnraw_gemmbk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
            make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, C), make_tuple(I0, I1));

            const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    wei_gemmbk0_gemmnraw_gemmbk1_grid_desc,
                    make_tuple(BK0, GemmNPerBlock, BK1),
                    Sequence<false, DoPadGemmN, false>{});

            return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            // GemmK is different for each GEMM
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // B weight tensor
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

            const auto wei_bk0_bk1_ydotslice_xdotslice_c_grid_desc =
                transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_slice_transform(YDot, I0, YDotSlice),
                                                       make_slice_transform(XDot, I0, XDotSlice),
                                                       make_freeze_transform(i_ytilde),
                                                       make_freeze_transform(i_xtilde),
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

            const auto wei_gemmbk0_gemmnraw_gemmbk1_grid_desc = transform_tensor_descriptor(
                wei_bk0_bk1_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, BK0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(BK1)),
                make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    wei_gemmbk0_gemmnraw_gemmbk1_grid_desc,
                    make_tuple(
                        wei_gemmbk0_gemmnraw_gemmbk1_grid_desc.GetLength(I0), GemmNPerBlock, BK1),
                    Sequence<false, DoPadGemmN, false>{});

            return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
        }
    }

    template <typename CLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          (is_same_v<CLayout, tensor_layout::convolution::GNHWC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::NHWGC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::G_NHW_C>),
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ytilde = tildes[0];
        index_t i_xtilde = tildes[1];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Hi = in_g_n_c_wis_lengths[3];
        const index_t Wi = in_g_n_c_wis_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        // assume strided
        const auto in_n_hi_wi_c_grid_desc =
            make_naive_tensor_descriptor(make_tuple(N, Hi, Wi, C),
                                         make_tuple(in_g_n_c_wis_strides[1],
                                                    in_g_n_c_wis_strides[3],
                                                    in_g_n_c_wis_strides[4],
                                                    in_g_n_c_wis_strides[2]));

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            // C: input tensor
            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                           make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_grid_desc,
                make_tuple(make_freeze_transform(I0),
                           make_freeze_transform(I0),
                           make_merge_transform(make_tuple(N, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<0, 2, 4>{}, Sequence<5>{}),
                make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmn_grid_desc = ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_gemmm_gemmn_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // C: input tensor
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
                           make_freeze_transform(i_ytilde),
                           make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                           make_freeze_transform(i_xtilde),
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

            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_n_htildeslice_wtildeslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmn_grid_desc = ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_gemmm_gemmn_grid_desc;
        }
    }

    // for input bias
    template <typename CLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          (is_same_v<CLayout, tensor_layout::convolution::GC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::G_C>),
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
                        const std::array<index_t, NDimSpatial>& /* tildes */)
    {
        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Hi = in_g_n_c_wis_lengths[3];
        const index_t Wi = in_g_n_c_wis_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const auto in_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, C), make_tuple(I0, I1));

            return in_gemmm_gemmn_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // bias tensor
            const auto in_gemmmraw_gemmnraw_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * HTildeSlice * WTildeSlice, C), make_tuple(I0, I1));

            const auto in_gemmm_gemmn_grid_desc = ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_gemmm_gemmn_grid_desc;
        }
    }
};

} // namespace tensor_operation
} // namespace ck
