// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardDataSpecialization ConvBackwardDataSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          typename M1N1ThreadClusterM1Xs,
          typename M1N1ThreadClusterN1Xs,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct DeviceConvNdBwdDataNwcKxcNwk_Dl
    : public DeviceConvBwdData<
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
          InDataType,
          WeiDataType,
          OutDataType,
          InElementwiseOperation,
          WeiElementwiseOperation,
          OutElementwiseOperation>
{
    using DeviceOp = DeviceConvNdBwdDataNwcKxcNwk_Dl;

    using ADataType = OutDataType;
    using BDataType = WeiDataType;
    using CDataType = InDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads,
                                                    std::vector<ck::index_t> tildes)
    {
        using namespace ck;

        index_t i_xtilde = tildes[0];

        const index_t Wi            = input_spatial_lengths[0];
        const index_t Wo            = output_spatial_lengths[0];
        const index_t X             = filter_spatial_lengths[0];
        const index_t InLeftPadW    = input_left_pads[0];
        const index_t InRightPadW   = input_right_pads[0];
        const index_t ConvStrideW   = conv_filter_strides[0];
        const index_t ConvDilationW = conv_filter_dilations[0];

        const auto K0 = K / K1;

        const auto in_n_wi_c_grid_desc = make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(N * Wo, K)),
                make_tuple(make_pass_through_transform(N * Wo),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            // B: weight tensor
            const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: input tensor
            const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_x_wo_c_grid_desc,
                make_tuple(make_freeze_transform(I0),
                           make_merge_transform(make_tuple(N, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
        }
        else
        {
            const auto out_n_wo_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wo, K));
            const auto wei_k_x_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, X, C));

            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // GemmK is different for each GEMM
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // A: output tensor
            const auto out_n_wop_k_grid_desc = transform_tensor_descriptor(
                out_n_wo_k_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wo, I0, I0),
                           make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto out_n_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                out_n_wop_k_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(XDot, WTilde),
                                         make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                    make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto out_n_xdotslice_wtildeslice_k0_k1_grid_desc = transform_tensor_descriptor(
                out_n_xdot_wtilde_k_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_slice_transform(XDot, I0, XDotSlice),
                           make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_n_xdotslice_wtildeslice_k0_k1_grid_desc,
                make_tuple(make_merge_transform(make_tuple(XDotSlice, K0)),
                           make_merge_transform(make_tuple(N, WTildeSlice)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<1, 3>{}, Sequence<0, 2>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // B weight tensor
            const auto wei_k_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                wei_k_x_c_grid_desc,
                make_tuple(make_pass_through_transform(K),
                           make_embed_transform(make_tuple(XDot, XTilde),
                                                make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto wei_k0_k1_xdotslice_c_grid_desc = transform_tensor_descriptor(
                wei_k_xdot_xtilde_c_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                           make_slice_transform(XDot, I0, XDotSlice),
                           make_freeze_transform(i_xtilde),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<>{}, Sequence<3>{}));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_k0_k1_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(XDotSlice, K0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<2, 0>{}, Sequence<3>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // C: input tensor
            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_xtilde_wtilde_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(XTilde, WTilde),
                                                make_tuple(ConvDilationW, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_n_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                in_n_xtilde_wtilde_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_freeze_transform(i_xtilde),
                           make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_wtildeslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, WTildeSlice)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
        }

    } // function end
    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads,
                                                    std::vector<ck::index_t> tildes)
    {
        using namespace ck;

        index_t i_ytilde = tildes[0];
        index_t i_xtilde = tildes[1];

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const auto K0 = K / K1;

        const auto out_n_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));
        const auto wei_k_y_x_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
        const auto in_n_hi_wi_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K)),
                make_tuple(make_pass_through_transform(N * Ho * Wo),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            // B: weight tensor
            const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: input tensor
            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                           make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_grid_desc,
                make_tuple(make_freeze_transform(I0),
                           make_freeze_transform(I0),
                           make_merge_transform(make_tuple(N, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<0, 2, 4>{}, Sequence<5>{}),
                make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
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

            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                           make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

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

            const auto wei_k0_k1_ydotslice_xdotslice_c_grid_desc =
                transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
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

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_k0_k1_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

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

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_htildeslice_wtildeslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
        }

    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads,
                                                    std::vector<ck::index_t> tildes)
    {
        using namespace ck;

        const index_t i_ztilde = tildes[0];
        const index_t i_ytilde = tildes[1];
        const index_t i_xtilde = tildes[2];

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t Z = filter_spatial_lengths[0];
        const index_t Y = filter_spatial_lengths[1];
        const index_t X = filter_spatial_lengths[2];

        const index_t InLeftPadD = input_left_pads[0];
        const index_t InLeftPadH = input_left_pads[1];
        const index_t InLeftPadW = input_left_pads[2];

        const index_t InRightPadD = input_right_pads[0];
        const index_t InRightPadH = input_right_pads[1];
        const index_t InRightPadW = input_right_pads[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        const index_t ConvDilationD = conv_filter_dilations[0];
        const index_t ConvDilationH = conv_filter_dilations[1];
        const index_t ConvDilationW = conv_filter_dilations[2];

        const auto K0 = K / K1;

        const auto out_n_do_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Do, Ho, Wo, K));
        const auto wei_k_z_y_x_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Z, Y, X, C));
        const auto in_n_di_hi_wi_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K)),
                make_tuple(make_pass_through_transform(N * Do * Ho * Wo),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            // B: weight tensor
            const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: input tensor
            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(I1, Do), make_tuple(I1, ConvStrideD)),
                           make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                           make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_freeze_transform(I0),
                           make_freeze_transform(I0),
                           make_freeze_transform(I0),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<1>{},
                           Sequence<3>{},
                           Sequence<5>{},
                           Sequence<0, 2, 4, 6>{},
                           Sequence<7>{}),
                make_tuple(Sequence<>{}, Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
        }
        else
        {
            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = ConvStrideD / GcdStrideDilationD;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto ZDot = math::integer_divide_ceil(Z, ZTilde);
            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto DTilde =
                Do + math::integer_divide_ceil(ConvDilationD * (Z - I1), ConvStrideD);
            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IDTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadD - ConvDilationD * (ZTilde - I1)), ConvStrideD);
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IDTildeSliceEnd = math::min(
                DTilde, math::integer_divide_ceil(InLeftPadD + Di - I1, ConvStrideD) + I1);
            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // GemmK is different for each GEMM
            const auto ZDotSlice = math::integer_divide_ceil(Z - i_ztilde, ZTilde);
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // A: output tensor
            const auto out_n_dop_hop_wop_k_grid_desc = transform_tensor_descriptor(
                out_n_do_ho_wo_k_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Do, I0, I0),
                           make_pad_transform(Ho, I0, I0),
                           make_pad_transform(Wo, I0, I0),
                           make_pass_through_transform(K)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc =
                transform_tensor_descriptor(
                    out_n_dop_hop_wop_k_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N),
                        make_embed_transform(make_tuple(ZDot, DTilde),
                                             make_tuple(-ConvDilationD / GcdStrideDilationD, I1)),
                        make_embed_transform(make_tuple(YDot, HTilde),
                                             make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                        make_embed_transform(make_tuple(XDot, WTilde),
                                             make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                        make_pass_through_transform(K)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

            const auto
                out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc =
                    transform_tensor_descriptor(
                        out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_slice_transform(ZDot, I0, ZDotSlice),
                                   make_slice_transform(DTilde, IDTildeSliceBegin, DTildeSlice),
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
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7, 8>{}));

            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc,
                make_tuple(
                    make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K0)),
                    make_merge_transform(make_tuple(N, DTildeSlice, HTildeSlice, WTildeSlice)),
                    make_pass_through_transform(K1)),
                make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}, Sequence<8>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // B weight tensor
            const auto wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc =
                transform_tensor_descriptor(
                    wei_k_z_y_x_c_grid_desc,
                    make_tuple(
                        make_pass_through_transform(K),
                        make_embed_transform(make_tuple(ZDot, ZTilde),
                                             make_tuple(ConvStrideD / GcdStrideDilationD, I1)),
                        make_embed_transform(make_tuple(YDot, YTilde),
                                             make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                        make_embed_transform(make_tuple(XDot, XTilde),
                                             make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                        make_pass_through_transform(C)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

            const auto wei_k0_k1_zdotslice_ydotslice_xdotslice_c_grid_desc =
                transform_tensor_descriptor(wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                       make_slice_transform(ZDot, I0, ZDotSlice),
                                                       make_slice_transform(YDot, I0, YDotSlice),
                                                       make_slice_transform(XDot, I0, XDotSlice),
                                                       make_freeze_transform(i_ztilde),
                                                       make_freeze_transform(i_ytilde),
                                                       make_freeze_transform(i_xtilde),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{},
                                                       Sequence<1>{},
                                                       Sequence<3>{},
                                                       Sequence<5>{},
                                                       Sequence<2>{},
                                                       Sequence<4>{},
                                                       Sequence<6>{},
                                                       Sequence<7>{}),
                                            make_tuple(Sequence<0, 1>{},
                                                       Sequence<2>{},
                                                       Sequence<3>{},
                                                       Sequence<4>{},
                                                       Sequence<>{},
                                                       Sequence<>{},
                                                       Sequence<>{},
                                                       Sequence<5>{}));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_k0_k1_zdotslice_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<2, 3, 4, 0>{}, Sequence<5>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // C: input tensor
            const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc =
                transform_tensor_descriptor(
                    in_n_dip_hip_wip_c_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_embed_transform(make_tuple(ZTilde, DTilde),
                                                    make_tuple(ConvDilationD, ConvStrideD)),
                               make_embed_transform(make_tuple(YTilde, HTilde),
                                                    make_tuple(ConvDilationH, ConvStrideH)),
                               make_embed_transform(make_tuple(XTilde, WTilde),
                                                    make_tuple(ConvDilationW, ConvStrideW)),
                               make_pass_through_transform(C)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

            const auto in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc =
                transform_tensor_descriptor(
                    in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_freeze_transform(i_ztilde),
                               make_slice_transform(DTilde, IDTildeSliceBegin, DTildeSlice),
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
                               Sequence<5>{},
                               Sequence<6>{},
                               Sequence<7>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<>{},
                               Sequence<1>{},
                               Sequence<>{},
                               Sequence<2>{},
                               Sequence<>{},
                               Sequence<3>{},
                               Sequence<4>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc,
                make_tuple(
                    make_merge_transform(make_tuple(N, DTildeSlice, HTildeSlice, WTildeSlice)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc);
        }

    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            1, 1, 1, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {0});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(1,
                                                                  1,
                                                                  1,
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {0, 0, 0});
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                     ADataType,
                                     AccDataType,
                                     CDataType,
                                     InMemoryDataOperationEnum::Set,
                                     AGridDesc_K0_M_K1,
                                     BGridDesc_K0_N_K1,
                                     CGridDesc_M_N,
                                     MPerBlock,
                                     NPerBlock,
                                     K0PerBlock,
                                     K1,
                                     M1PerThread,
                                     N1PerThread,
                                     KPerThread,
                                     M1N1ThreadClusterM1Xs,
                                     M1N1ThreadClusterN1Xs,
                                     ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterArrangeOrder,
                                     ABlockTransferSrcAccessOrder,
                                     ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                     ABlockTransferSrcVectorTensorContiguousDimOrder,
                                     ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                     BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                     BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                     BBlockTransferThreadClusterArrangeOrder,
                                     BBlockTransferSrcAccessOrder,
                                     BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                     BBlockTransferSrcVectorTensorContiguousDimOrder,
                                     BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                     CThreadTransferSrcDstAccessOrder,
                                     CThreadTransferSrcDstVectorDim,
                                     CThreadTransferDstScalarPerVector>;

    using AGridDesc_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{}));
    using DefaultBlock2CTileMap =
        decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}));
    // Argument
    struct Argument : public BaseArgument
    {
        Argument(InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t K,
                 ck::index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_in_grid},
              a_element_op_{out_element_op},
              b_element_op_{wei_element_op},
              c_element_op_{in_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              input_spatial_lengths_{input_spatial_lengths},
              filter_spatial_lengths_{filter_spatial_lengths},
              output_spatial_lengths_{output_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            CreateABCDesc<NDimSpatial>();
        }

        template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
        void CreateABCDesc()
        {
            const index_t ConvStrideW     = conv_filter_strides_[0];
            const index_t ConvDilationW   = conv_filter_dilations_[0];
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);
            const auto XTilde             = ConvStrideW / GcdStrideDilationW;

            const index_t X = filter_spatial_lengths_[0];

            for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
            {
                // check slice is valid
                const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);
                if(XDotSlice <= 0)
                {
                    continue;
                }

                const auto descs =
                    DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        conv_filter_strides_,
                        conv_filter_dilations_,
                        input_left_pads_,
                        input_right_pads_,
                        {i_xtilde});
                a_grid_desc_k0_m_k1_container_.push_back(descs[I0]);
                b_grid_desc_k0_n_k1_container_.push_back(descs[I1]);
                c_grid_desc_m_n_container_.push_back(descs[I2]);

                if(GridwiseGemm::CheckValidity(descs[I0], descs[I1], descs[I2]))
                {
                    a_grid_desc_k0_m0_m1_k1_container_.push_back(
                        GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(descs[I0]));
                    b_grid_desc_k0_n0_n1_k1_container_.push_back(
                        GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(descs[I1]));
                    c_grid_desc_m0_m10_m11_n0_n10_n11_container_.push_back(
                        GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(descs[I2]));

                    block_2_ctile_map_container_.push_back(
                        GridwiseGemm::MakeDefaultBlock2CTileMap(descs[I2]));
                }
            }
        }
        template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
        void CreateABCDesc()
        {
            const index_t ConvStrideH = conv_filter_strides_[0];
            const index_t ConvStrideW = conv_filter_strides_[1];

            const index_t ConvDilationH = conv_filter_dilations_[0];
            const index_t ConvDilationW = conv_filter_dilations_[1];

            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const index_t Y = filter_spatial_lengths_[0];
            const index_t X = filter_spatial_lengths_[1];
            for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
            {
                for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                {
                    // check slice is valid
                    const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                    const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);
                    if(YDotSlice * XDotSlice <= 0)
                    {
                        continue;
                    }

                    const auto descs =
                        DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                            Conv_N_,
                            Conv_K_,
                            Conv_C_,
                            input_spatial_lengths_,
                            filter_spatial_lengths_,
                            output_spatial_lengths_,
                            conv_filter_strides_,
                            conv_filter_dilations_,
                            input_left_pads_,
                            input_right_pads_,
                            {i_ytilde, i_xtilde});
                    a_grid_desc_k0_m_k1_container_.push_back(descs[I0]);
                    b_grid_desc_k0_n_k1_container_.push_back(descs[I1]);
                    c_grid_desc_m_n_container_.push_back(descs[I2]);

                    if(GridwiseGemm::CheckValidity(descs[I0], descs[I1], descs[I2]))
                    {
                        a_grid_desc_k0_m0_m1_k1_container_.push_back(
                            GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(descs[I0]));
                        b_grid_desc_k0_n0_n1_k1_container_.push_back(
                            GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(descs[I1]));
                        c_grid_desc_m0_m10_m11_n0_n10_n11_container_.push_back(
                            GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(descs[I2]));

                        block_2_ctile_map_container_.push_back(
                            GridwiseGemm::MakeDefaultBlock2CTileMap(descs[I2]));
                    }
                }
            }
        }
        template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
        void CreateABCDesc()
        {
            const index_t ConvStrideD = conv_filter_strides_[0];
            const index_t ConvStrideH = conv_filter_strides_[1];
            const index_t ConvStrideW = conv_filter_strides_[2];

            const index_t ConvDilationD = conv_filter_dilations_[0];
            const index_t ConvDilationH = conv_filter_dilations_[1];
            const index_t ConvDilationW = conv_filter_dilations_[2];

            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = ConvStrideD / GcdStrideDilationD;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const index_t Z = filter_spatial_lengths_[0];
            const index_t Y = filter_spatial_lengths_[1];
            const index_t X = filter_spatial_lengths_[2];
            for(index_t i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
            {
                for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
                {
                    for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                    {
                        // check slice is valid
                        const auto ZDotSlice = math::integer_divide_ceil(Z - i_ztilde, ZTilde);
                        const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                        const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);
                        if(ZDotSlice * YDotSlice * XDotSlice <= 0)
                        {
                            continue;
                        }

                        const auto descs =
                            DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                                Conv_N_,
                                Conv_K_,
                                Conv_C_,
                                input_spatial_lengths_,
                                filter_spatial_lengths_,
                                output_spatial_lengths_,
                                conv_filter_strides_,
                                conv_filter_dilations_,
                                input_left_pads_,
                                input_right_pads_,
                                {i_ztilde, i_ytilde, i_xtilde});
                        a_grid_desc_k0_m_k1_container_.push_back(descs[I0]);
                        b_grid_desc_k0_n_k1_container_.push_back(descs[I1]);
                        c_grid_desc_m_n_container_.push_back(descs[I2]);

                        if(GridwiseGemm::CheckValidity(descs[I0], descs[I1], descs[I2]))
                        {
                            a_grid_desc_k0_m0_m1_k1_container_.push_back(
                                GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(descs[I0]));
                            b_grid_desc_k0_n0_n1_k1_container_.push_back(
                                GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(descs[I1]));
                            c_grid_desc_m0_m10_m11_n0_n10_n11_container_.push_back(
                                GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(descs[I2]));

                            block_2_ctile_map_container_.push_back(
                                GridwiseGemm::MakeDefaultBlock2CTileMap(descs[I2]));
                        }
                    }
                }
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        std::vector<AGridDesc_K0_M_K1> a_grid_desc_k0_m_k1_container_;
        std::vector<BGridDesc_K0_N_K1> b_grid_desc_k0_n_k1_container_;
        std::vector<CGridDesc_M_N> c_grid_desc_m_n_container_;

        std::vector<AGridDesc_K0_M0_M1_K1> a_grid_desc_k0_m0_m1_k1_container_;
        std::vector<BGridDesc_K0_N0_N1_K1> b_grid_desc_k0_n0_n1_k1_container_;
        std::vector<CGridDesc_M0_M10_M11_N0_N10_N11> c_grid_desc_m0_m10_m11_n0_n10_n11_container_;

        std::vector<DefaultBlock2CTileMap> block_2_ctile_map_container_;

        // element-wise op
        OutElementwiseOperation a_element_op_;
        WeiElementwiseOperation b_element_op_;
        InElementwiseOperation c_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;

        std::vector<ck::index_t> input_spatial_lengths_;
        std::vector<ck::index_t> filter_spatial_lengths_;
        std::vector<ck::index_t> output_spatial_lengths_;
        std::vector<ck::index_t> conv_filter_strides_;
        std::vector<ck::index_t> conv_filter_dilations_;
        std::vector<ck::index_t> input_left_pads_;
        std::vector<ck::index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float ave_time = 0;
            for(size_t i = 0; i < arg.a_grid_desc_k0_m_k1_container_.size(); i++)
            {
#if DEBUG_LOG
                {
                    std::cout << "arg.a_grid_desc_k0_m_k1_container_{"
                              << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I0) << ", "
                              << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I1) << ", "
                              << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I2) << "}"
                              << std::endl;

                    std::cout << "arg.b_grid_desc_k0_n_k1_container_{"
                              << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I0) << ", "
                              << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I1) << ", "
                              << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I2) << "}"
                              << std::endl;

                    std::cout << "arg.c_grid_desc_m_n_container_{ "
                              << arg.c_grid_desc_m_n_container_[i].GetLength(I0) << ", "
                              << arg.c_grid_desc_m_n_container_[i].GetLength(I1) << "}"
                              << std::endl;

                    std::cout << "arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_( "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I0)
                              << ", "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I1)
                              << ", "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I2)
                              << ", "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I3)
                              << ", "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I4)
                              << ", "
                              << arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i].GetLength(I5)
                              << " ) " << std::endl;
                }
#endif

                if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_container_[i],
                                                arg.b_grid_desc_k0_n_k1_container_[i],
                                                arg.c_grid_desc_m_n_container_[i]))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
                }

                const index_t grid_size = arg.block_2_ctile_map_container_[i].CalculateGridSize(
                    arg.c_grid_desc_m_n_container_[i]);

                auto launch_kernel = [&](auto has_main_k_block_loop,
                                         auto has_double_tail_k_block_loop) {
                    constexpr bool has_main_loop   = has_main_k_block_loop.value;
                    constexpr bool has_double_loop = has_double_tail_k_block_loop;

                    const auto kernel = kernel_gemm_dl_v1r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M0_M1_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N0_N1_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_M0_M10_M11_N0_N10_N11>,
                        remove_reference_t<DeviceOp::DefaultBlock2CTileMap>,
                        has_main_loop,
                        has_double_loop>;

                    ave_time +=
                        launch_and_time_kernel(stream_config,
                                               kernel,
                                               dim3(grid_size),
                                               dim3(BlockSize),
                                               0,
                                               arg.p_a_grid_,
                                               arg.p_b_grid_,
                                               arg.p_c_grid_,
                                               arg.a_grid_desc_k0_m0_m1_k1_container_[i],
                                               arg.b_grid_desc_k0_n0_n1_k1_container_[i],
                                               arg.c_grid_desc_m0_m10_m11_n0_n10_n11_container_[i],
                                               arg.block_2_ctile_map_container_[i]);
                };

                const auto K0 = arg.a_grid_desc_k0_m0_m1_k1_container_[i].GetLength(I0);
                const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
                const bool has_double_tail_k_block_loop =
                    GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    launch_kernel(integral_constant<bool, true>{}, integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    launch_kernel(integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    launch_kernel(integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    launch_kernel(integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }
            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        // check device
        if(!(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030"))
        {
            return false;
        }

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // matrix A
        {
            auto srcVectorLengths = ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1{};
            if(srcVectorLengths[I1] != 1 || srcVectorLengths[I2] != 1)
            {
                return false;
            }
            if(K1 % srcVectorLengths[I3] != 0 || K0PerBlock % srcVectorLengths[I0] != 0)
            {
                return false;
            }

            const index_t K = arg.Conv_K_;

            if(K % (srcVectorLengths[I0] * srcVectorLengths[I3]) != 0)
            {
                return false;
            }
        }

        // matrix B
        {
            auto srcLoadLenghts   = BBlockTransferThreadSliceLengths_K0_N0_N1_K1{};
            auto srcVectorLengths = BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1{};
            if(srcVectorLengths[I0] != 1 || srcVectorLengths[I3] != 1)
            {
                return false;
            }
            if(srcLoadLenghts[I1] % srcVectorLengths[I1] != 0 ||
               srcLoadLenghts[I2] % srcVectorLengths[I2] != 0)
            {
                return false;
            }

            const index_t C = arg.Conv_K_;

            if(C % (srcVectorLengths[I1] * srcVectorLengths[I2]) != 0)
            {
                return false;
            }
        }
        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CThreadTransferDstScalarPerVector == 0))
        {
            std::cout << "Not surpport,because: arg.Conv_C_ % CThreadTransferDstScalarPerVector = "
                      << arg.Conv_C_ % CThreadTransferDstScalarPerVector << std::endl;
            return false;
        }

        // Gridwise GEMM size
        for(std::size_t i = 0; i < arg.a_grid_desc_k0_m_k1_container_.size(); i++)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_container_[i],
                                            arg.b_grid_desc_k0_n_k1_container_[i],
                                            arg.c_grid_desc_m_n_container_[i]))
            {
                return false;
            }
        }
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             const OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_in_grid,
                        const void* p_wei_grid,
                        const void* p_out_grid,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(static_cast<InDataType*>(p_in_grid),
                                          static_cast<const WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConvNdBwdDataNwcKxcNwk_Dl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0){

            str<< " Filter1x1Stride1Pad0";
        }


        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
