// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] =
//     activate(in[N, Hi, Wi, C] * wei[K, Y, X, C] + bias[K])
template <
    typename InDataType,
    typename WeiDataType,
    typename OutDataType,
    typename AccDataType,
    typename InElementwiseOperation,
    typename WeiElementwiseOperation,
    typename OutElementwiseOperation,
    InMemoryDataOperationEnum OutGlobalMemoryDataOperation,
    ConvolutionForwardSpecialization ConvForwardSpecialization,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t K0PerBlock,
    ck::index_t K1,
    ck::index_t MPerXDL,
    ck::index_t NPerXDL,
    ck::index_t MXdlPerWave,
    ck::index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_K0_M_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    ck::index_t ABlockTransferSrcVectorDim,
    ck::index_t ABlockTransferSrcScalarPerVector,
    ck::index_t ABlockTransferDstScalarPerVector_K1,
    bool ABlockLdsAddExtraM,
    typename BBlockTransferThreadClusterLengths_K0_N_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    ck::index_t BBlockTransferSrcVectorDim,
    ck::index_t BBlockTransferSrcScalarPerVector,
    ck::index_t BBlockTransferDstScalarPerVector_K1,
    bool BBlockLdsAddExtraN,
    index_t CShuffleMXdlPerWavePerShuffle,
    index_t CShuffleNXdlPerWavePerShuffle,
    typename CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
    index_t CBlockTransferScalarPerVector_NWaveNPerXdl>
struct DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
    : public DeviceConvFwdBiasActivation<InElementwiseOperation,
                                         WeiElementwiseOperation,
                                         OutElementwiseOperation>
{
    using DeviceOp =
        DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    // TODO make it support any # of spatial dimensions
    static constexpr index_t NDimSpatial = 2;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

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
                                                    std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t GemmMRaw = N * Ho * Wo;
        const index_t GemmN    = K;

        const auto GemmM    = math::integer_least_multiple(GemmMRaw, MPerBlock);
        const auto GemmMPad = GemmM - GemmMRaw;

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        { // 1x1, stride=1, pad=0
            const index_t GemmK = Y * X * C;
            assert(GemmK % GemmK1Number == 0);

            const index_t GemmK0 = GemmK / GemmK1Number;

            // A: input tensor
            const auto in_gemmmraw_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, C));

            const auto in_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmmraw_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmMRaw, GemmMPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: weight tensor
            const auto wei_gemmn_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, C));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_gemmn_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: output tensor
            const auto out_gemmmraw_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

            const auto out_gemmm_gemmn_grid_desc =
                transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmMRaw, GemmMPad),
                                                       make_pass_through_transform(GemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            // C0: bias tensor: assume a contiguous vector
            const auto bias_grid_desc_gemmm_gemmn =
                make_naive_tensor_descriptor(make_tuple(GemmM, GemmN), make_tuple(I0, I1));

            return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              out_gemmm_gemmn_grid_desc,
                              bias_grid_desc_gemmm_gemmn);
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        { // 1x1, pad=0
            const index_t GemmK = Y * X * C;
            assert(GemmK % GemmK1Number == 0);

            const index_t GemmK0 = GemmK / GemmK1Number;

            // A: input tensor
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_ho_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_n_ho_wo_c_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_merge_transform(make_tuple(N, Ho, Wo))),
                make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(GemmK0),
                           make_right_pad_transform(GemmMRaw, GemmMPad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // B: weight tensor
            const auto wei_gemmn_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, C));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_gemmn_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: output tensor
            const auto out_gemmmraw_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

            const auto out_gemmm_gemmn_grid_desc =
                transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmMRaw, GemmMPad),
                                                       make_pass_through_transform(GemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            // C0: bias tensor: assume a contiguous vector
            const auto bias_grid_desc_gemmm_gemmn =
                make_naive_tensor_descriptor(make_tuple(GemmM, GemmN), make_tuple(I0, I1));

            return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              out_gemmm_gemmn_grid_desc,
                              bias_grid_desc_gemmm_gemmn);
        }
        else if constexpr(ConvForwardSpecialization == ConvolutionForwardSpecialization::OddC)
        { // C = odd value
            const index_t GemmKRaw = Y * X * C;
            const index_t GemmK = math::integer_least_multiple(GemmKRaw, K0PerBlock * GemmK1Number);
            const index_t GemmKPad = GemmK - GemmKRaw;
            const index_t GemmK0   = GemmK / GemmK1Number;

            // A: input tensor
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmkraw_gemmmraw_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk_gemmm_grid_desc = transform_tensor_descriptor(
                in_gemmkraw_gemmmraw_grid_desc,
                make_tuple(make_right_pad_transform(GemmKRaw, GemmKPad),
                           make_right_pad_transform(GemmMRaw, GemmMPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: weight tensor
            const auto wei_k_yxc_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

            const auto wei_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
                wei_k_yxc_grid_desc,
                make_tuple(make_pass_through_transform(K),
                           make_right_pad_transform(GemmKRaw, GemmKPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_gemmk_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: output tensor
            const auto out_nhowo_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

            const auto out_gemmmraw_gemmn_grid_desc =
                transform_tensor_descriptor(out_nhowo_k_grid_desc,
                                            make_tuple(make_pass_through_transform(N * Ho * Wo),
                                                       make_pass_through_transform(K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmm_gemmn_grid_desc =
                transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmMRaw, GemmMPad),
                                                       make_pass_through_transform(GemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            // C0: bias tensor: assume a contiguous vector
            const auto bias_grid_desc_gemmm_gemmn =
                make_naive_tensor_descriptor(make_tuple(GemmM, GemmN), make_tuple(I0, I1));

            return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              out_gemmm_gemmn_grid_desc,
                              bias_grid_desc_gemmm_gemmn);
        }
        else
        {
            const index_t GemmK = Y * X * C;
            assert(GemmK % GemmK1Number == 0);

            const index_t GemmK0 = GemmK / GemmK1Number;

            // A: input tensor
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmk_gemmmraw_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk_gemmmraw_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmMRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(GemmK0),
                           make_right_pad_transform(GemmMRaw, GemmMPad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // B: weight tensor
            const auto wei_k_yxc_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

            const auto wei_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
                wei_k_yxc_grid_desc,
                make_tuple(make_pass_through_transform(K), make_pass_through_transform(Y * X * C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_gemmk_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: output tensor
            const auto out_nhowo_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

            const auto out_gemmmraw_gemmn_grid_desc =
                transform_tensor_descriptor(out_nhowo_k_grid_desc,
                                            make_tuple(make_pass_through_transform(N * Ho * Wo),
                                                       make_pass_through_transform(K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmm_gemmn_grid_desc =
                transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmMRaw, GemmMPad),
                                                       make_pass_through_transform(GemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            // C0: bias tensor: assume a contiguous vector
            const auto bias_grid_desc_gemmm_gemmn =
                make_naive_tensor_descriptor(make_tuple(GemmM, GemmN), make_tuple(I0, I1));

            return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              out_gemmm_gemmn_grid_desc,
                              bias_grid_desc_gemmm_gemmn);
        }
    }

    using ABCGridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}));

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;
    using C0GridDesc_M_N    = remove_cvref_t<decltype(ABCGridDescs{}[I3])>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r2<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        OutGlobalMemoryDataOperation,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        C0GridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,                 // ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // BBlockTransferSrcAccessOrder,
        2,                 // BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
        CBlockTransferScalarPerVector_NWaveNPerXdl>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 OutDataType* p_out_grid,
                 const OutDataType* p_bias_grid,
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
                 ck::index_t M01,
                 ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              p_c0_grid_{p_bias_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c0_grid_desc_m_n_{},
              c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_{},
              c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op},
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
            const auto descs =
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(N,
                                                                          K,
                                                                          C,
                                                                          input_spatial_lengths,
                                                                          filter_spatial_lengths,
                                                                          output_spatial_lengths,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads);

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];
            c0_grid_desc_m_n_    = descs[I3];
            block_2_ctile_map_ =
                GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);

            if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                           b_grid_desc_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_ =
                    GridwiseGemm::
                        MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                            c_grid_desc_m_n_);

                c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_ =
                    GridwiseGemm::
                        MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                            c0_grid_desc_m_n_);
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        const CDataType* p_c0_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        C0GridDesc_M_N c0_grid_desc_m_n_;
        typename GridwiseGemm::
            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
                c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_;
        typename GridwiseGemm::
            C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
                c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::vector<index_t> input_spatial_lengths_;
        std::vector<index_t> filter_spatial_lengths_;
        std::vector<index_t> output_spatial_lengths_;
        std::vector<index_t> conv_filter_strides_;
        std::vector<index_t> conv_filter_dilations_;
        std::vector<index_t> input_left_pads_;
        std::vector<index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if DEBUG_LOG
            {
                std::cout << DeviceOp{}.GetTypeString() << std::endl;
                std::cout << "N " << arg.Conv_N_ << ", "
                          << "K " << arg.Conv_K_ << ", "
                          << "C " << arg.Conv_C_ << ", " << std::endl;
                std::cout << "Y X " << arg.filter_spatial_lengths_[0] << ", "
                          << arg.filter_spatial_lengths_[1] << ", " << std::endl;
                std::cout << "Hi Wi " << arg.input_spatial_lengths_[0] << ", "
                          << arg.input_spatial_lengths_[1] << ", " << std::endl;
                std::cout << "Ho Wo " << arg.output_spatial_lengths_[0] << ", "
                          << arg.output_spatial_lengths_[1] << ", " << std::endl;
                std::cout << "Strides " << arg.conv_filter_strides_[0] << ", "
                          << arg.conv_filter_strides_[1] << ", " << std::endl;
                std::cout << "Dilations " << arg.conv_filter_dilations_[0] << ", "
                          << arg.conv_filter_dilations_[1] << ", " << std::endl;
                std::cout << "InLeftPads " << arg.input_left_pads_[0] << ", "
                          << arg.input_left_pads_[1] << ", " << std::endl;
                std::cout << "InLeftPads " << arg.input_right_pads_[0] << ", "
                          << arg.input_right_pads_[1] << ", " << std::endl;
            }

            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                std::cout << "arg.c0_grid_desc_m_n_{ " << arg.c0_grid_desc_m_n_.GetLength(I0)
                          << ", " << arg.c0_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r2 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_xdlops_v3r2<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a_grid_,
                    arg.p_b_grid_,
                    arg.p_c_grid_,
                    arg.p_c0_grid_,
                    arg.a_grid_desc_k0_m_k1_,
                    arg.b_grid_desc_k0_n_k1_,
                    arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
                    arg.c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
                    arg.in_element_op_,
                    arg.wei_element_op_,
                    arg.out_element_op_,
                    arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v3r2<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a_grid_,
                    arg.p_b_grid_,
                    arg.p_c_grid_,
                    arg.p_c0_grid_,
                    arg.a_grid_desc_k0_m_k1_,
                    arg.b_grid_desc_k0_n_k1_,
                    arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
                    arg.c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
                    arg.in_element_op_,
                    arg.wei_element_op_,
                    arg.out_element_op_,
                    arg.block_2_ctile_map_);
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
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            if(!(arg.filter_spatial_lengths_[0] == 1 && arg.filter_spatial_lengths_[1] == 1 &&
                 arg.conv_filter_strides_[0] == 1 && arg.conv_filter_strides_[1] == 1 &&
                 arg.input_left_pads_[0] == 0 && arg.input_left_pads_[1] == 0 &&
                 arg.input_right_pads_[0] == 0 && arg.input_right_pads_[1] == 0))
            {
                return false;
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            if(!(arg.filter_spatial_lengths_[0] == 1 && arg.filter_spatial_lengths_[1] == 1 &&
                 arg.input_left_pads_[0] == 0 && arg.input_left_pads_[1] == 0 &&
                 arg.input_right_pads_[0] == 0 && arg.input_right_pads_[1] == 0))
            {
                return false;
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_C_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_K_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             OutDataType* p_out_grid,
                             const OutDataType* p_bias_grid,
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
                        p_bias_grid,
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
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        const void* p_wei_grid,
                        void* p_out_grid,
                        const void* p_bias_grid,
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
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<const WeiDataType*>(p_wei_grid),
                                          static_cast<OutDataType*>(p_out_grid),
                                          static_cast<const OutDataType*>(p_bias_grid),
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
                                          1,
                                          1,
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
        str << "DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
