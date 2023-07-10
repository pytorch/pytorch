// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXdl,
          ck::index_t NPerXdl,
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
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl>
struct DeviceConv2dBwdWeightXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
    : public DeviceConvBwdWeight<2,
                                 ck::tensor_layout::convolution::NHWC,
                                 ck::tensor_layout::convolution::KYXC,
                                 ck::tensor_layout::convolution::NHWK,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 InElementwiseOperation,
                                 WeiElementwiseOperation,
                                 OutElementwiseOperation>
{
    static constexpr ck::index_t NDimSpatial = 2;

    using DeviceOp =
        DeviceConv2dBwdWeightXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K;

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

    using AElementwiseOperation = OutElementwiseOperation;
    using BElementwiseOperation = InElementwiseOperation;
    using CElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    static constexpr auto N1Number = K1Number;

    // Bytes per 32 lds bank: 32 * 4 bytes
    static constexpr auto BankLength = 128;
    static constexpr auto ElePerBank = BankLength / sizeof(ADataType);

    // M1 & M0
    static constexpr auto ABlockLdsM1PerBlock = ElePerBank / K1;
    static constexpr auto ABlockLdsM0PerBlock = MPerBlock / ABlockLdsM1PerBlock;
    static constexpr auto ABlockLdsM1Padding  = 4;

    // N1 & N0
    static constexpr auto BBlockLdsN1PerBlock = ElePerBank / K1;
    static constexpr auto BBlockLdsN0PerBlock = NPerBlock / BBlockLdsN1PerBlock;
    static constexpr auto BBlockLdsN1Padding  = 4;

    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        ck::index_t N,
        ck::index_t K,
        ck::index_t C,
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths,
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths,
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths,
        std::array<ck::index_t, NDimSpatial> conv_filter_strides,
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations,
        std::array<ck::index_t, NDimSpatial> input_left_pads,
        std::array<ck::index_t, NDimSpatial> input_right_pads,
        ck::index_t batch_k)
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

        const index_t GemmKTotal = N * Ho * Wo;
        const index_t GemmM      = K;
        const index_t GemmN      = C * X * Y;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;

        const auto in_n_hi_wi_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

        // A: output tensor
        const index_t N0          = N / N1Number;
        const index_t GemmK0Total = N0 * Ho * Wo;

        const index_t GemmK0S =
            math::integer_divide_ceil(GemmK0Total, K0PerBlock * GemmKBatch) * K0PerBlock;
        const index_t GemmK0Pad = GemmKBatch * GemmK0S;
        const auto out_n_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Ho * Wo, K));

        const auto out_n0_ho_wo_k_n1_grid_desc =
            transform_tensor_descriptor(out_n_ho_wo_k_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(N0, N1Number)),
                                                   make_pass_through_transform(Ho * Wo),
                                                   make_pass_through_transform(K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}));

        const auto out_gemmk0total_gemmm_gemmk1_grid_desc =
            transform_tensor_descriptor(out_n0_ho_wo_k_n1_grid_desc,
                                        make_tuple(make_merge_transform(make_tuple(N0, Ho * Wo)),
                                                   make_pass_through_transform(K),
                                                   make_pass_through_transform(N1Number)),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto out_gemmk0pad_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
            out_gemmk0total_gemmm_gemmk1_grid_desc,
            make_tuple(make_right_pad_transform(GemmK0Total, GemmK0Pad - GemmK0Total),
                       make_pass_through_transform(GemmM),
                       make_pass_through_transform(N1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
            out_gemmk0pad_gemmm_gemmk1_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0)),
                       make_pass_through_transform(GemmM),
                       make_pass_through_transform(N1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // B: input tensor
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

        const auto in_n0_y_ho_x_wo_c_n1_grid_desc =
            transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(N0, N1Number)),
                                                   make_pass_through_transform(Y),
                                                   make_pass_through_transform(Ho),
                                                   make_pass_through_transform(X),
                                                   make_pass_through_transform(Wo),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{}),
                                        make_tuple(Sequence<0, 6>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{}));

        const auto in_gemmk0total_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
            in_n0_y_ho_x_wo_c_n1_grid_desc,
            make_tuple(make_merge_transform(make_tuple(N0, Ho, Wo)),
                       make_merge_transform(make_tuple(Y, X, C)),
                       make_pass_through_transform(N1Number)),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto in_gemmk0pad_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
            in_gemmk0total_gemmn_gemmk1_grid_desc,
            make_tuple(make_right_pad_transform(GemmK0Total, GemmK0Pad - GemmK0Total),
                       make_pass_through_transform(GemmN),
                       make_pass_through_transform(N1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
            in_gemmk0pad_gemmn_gemmk1_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0)),
                       make_pass_through_transform(GemmN),
                       make_pass_through_transform(N1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // C: weight tensor
        const auto wei_gemmm_gemmn_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

        return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                          in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                          wei_gemmm_gemmn_grid_desc);
    }

    using ABCGridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 1));

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXdl,
        NPerXdl,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        ABlockLdsM1PerBlock,
        ABlockLdsM0PerBlock,
        ABlockLdsM1Padding,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        BBlockLdsN1PerBlock,
        BBlockLdsN0PerBlock,
        BBlockLdsN1Padding,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true,
        true>;

    using GridwiseGemmAtomicAdd = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXdl,
        NPerXdl,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        ABlockLdsM1PerBlock,
        ABlockLdsM0PerBlock,
        ABlockLdsM1Padding,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        BBlockLdsN1PerBlock,
        BBlockLdsN0PerBlock,
        BBlockLdsN1Padding,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true,
        true>;
    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));

    using Block2CTileMap =
        decltype(GridwiseGemm::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t K,
                 ck::index_t C,
                 std::array<ck::index_t, NDimSpatial> input_spatial_lengths,
                 std::array<ck::index_t, NDimSpatial> filter_spatial_lengths,
                 std::array<ck::index_t, NDimSpatial> output_spatial_lengths,
                 std::array<ck::index_t, NDimSpatial> conv_filter_strides,
                 std::array<ck::index_t, NDimSpatial> conv_filter_dilations,
                 std::array<ck::index_t, NDimSpatial> input_left_pads,
                 std::array<ck::index_t, NDimSpatial> input_right_pads,
                 ck::index_t M01,
                 ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_c_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              output_spatial_lengths_{output_spatial_lengths},
              filter_spatial_lengths_{filter_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              k_batch_{split_k}
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
                                                                          input_right_pads,
                                                                          k_batch_);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            block_2_ctile_map_ =
                GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            if(GridwiseGemm::CheckValidity(a_grid_desc_kbatch_k0_m_k1_,
                                           b_grid_desc_kbatch_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n_);
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation a_element_op_;
        OutElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::array<index_t, NDimSpatial> output_spatial_lengths_;
        std::array<index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void Print(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(arg);
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight has invalid setting");
            }
            const auto kbatch = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0);
            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                hipGetErrorString(hipMemset(
                    arg.p_c_grid_,
                    0,
                    arg.c_grid_desc_mblock_mperblock_nblock_nperblock_.GetElementSpaceSize() *
                        sizeof(CDataType)));

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.block_2_ctile_map_);
            };

            if(has_main_k0_block_loop)
            {
                if(kbatch == 1)
                {
                    const auto kernel = kernel_gemm_xdlops_bwd_weight<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        OutElementwiseOperation,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        remove_reference_t<DeviceOp::Block2CTileMap>,
                        true>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdlops_bwd_weight<
                        GridwiseGemmAtomicAdd,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        OutElementwiseOperation,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        remove_reference_t<DeviceOp::Block2CTileMap>,
                        true>;

                    Run(kernel);
                }
            }
            else
            {
                if(kbatch == 1)
                {
                    const auto kernel = kernel_gemm_xdlops_bwd_weight<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        OutElementwiseOperation,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        remove_reference_t<DeviceOp::Block2CTileMap>,
                        false>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdlops_bwd_weight<
                        GridwiseGemmAtomicAdd,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        OutElementwiseOperation,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        remove_reference_t<DeviceOp::Block2CTileMap>,
                        false>;

                    Run(kernel);
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
        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // unmerge N to N0 and N1, where N1 equals to K1
        if(!(arg.Conv_N_ % K1 == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             WeiDataType* p_wei_grid,
                             const OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t C,
                             std::array<ck::index_t, NDimSpatial> input_spatial_lengths,
                             std::array<ck::index_t, NDimSpatial> filter_spatial_lengths,
                             std::array<ck::index_t, NDimSpatial> output_spatial_lengths,
                             std::array<ck::index_t, NDimSpatial> conv_filter_strides,
                             std::array<ck::index_t, NDimSpatial> conv_filter_dilations,
                             std::array<ck::index_t, NDimSpatial> input_left_pads,
                             std::array<ck::index_t, NDimSpatial> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op,
                             ck::index_t split_k)
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
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::array<ck::index_t, NDimSpatial> input_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> output_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> conv_filter_strides,
                        std::array<ck::index_t, NDimSpatial> conv_filter_dilations,
                        std::array<ck::index_t, NDimSpatial> input_left_pads,
                        std::array<ck::index_t, NDimSpatial> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
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
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv2dBwdWeightXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K"
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
