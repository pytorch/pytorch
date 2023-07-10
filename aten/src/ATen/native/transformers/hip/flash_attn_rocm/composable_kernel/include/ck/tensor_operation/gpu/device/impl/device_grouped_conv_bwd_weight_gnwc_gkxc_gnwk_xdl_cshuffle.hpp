// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

struct ComputePtrOffsetOfStridedBatch
{
    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideA_);
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideB_);
    }

    __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideC_);
    }

    index_t BatchStrideA_;
    index_t BatchStrideB_;
    index_t BatchStrideC_;
};

} // namespace

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_B_K0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_bwd_weight(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const index_t batch_count,
            const AGridDesc_B_K0_M_K1 a_b_k0_m_k1_grid_desc,
            const BGridDesc_B_K0_N_K1 b_b_k0_n_k1_grid_desc,
            const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            const Block2CTileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

    __shared__ FloatAB p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB)];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_c_grid + c_batch_offset,
                                                  p_shared,
                                                  a_b_k0_m_k1_grid_desc,
                                                  b_b_k0_n_k1_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_b_k0_m_k1_grid_desc;
    ignore = b_b_k0_n_k1_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = batch_count;
    ignore = block_2_ctile_map;
    ignore = compute_ptr_offset_of_batch;

    compute_ptr_offset_of_batch.GetAPtrOffset(0);
    compute_ptr_offset_of_batch.GetBPtrOffset(0);
    compute_ptr_offset_of_batch.GetCPtrOffset(0);
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
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
struct DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle
    : public DeviceGroupedConvBwdWeight<
          NDimSpatial,
          ck::tuple_element_t<NDimSpatial - 1,
                              ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                        ck::tensor_layout::convolution::GNHWC,
                                        ck::tensor_layout::convolution::GNDHWC>>,
          ck::tuple_element_t<NDimSpatial - 1,
                              ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                        ck::tensor_layout::convolution::GKYXC,
                                        ck::tensor_layout::convolution::GKZYXC>>,
          ck::tuple_element_t<NDimSpatial - 1,
                              ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                        ck::tensor_layout::convolution::GNHWK,
                                        ck::tensor_layout::convolution::GNDHWK>>,
          InDataType,
          WeiDataType,
          OutDataType,
          InElementwiseOperation,
          WeiElementwiseOperation,
          OutElementwiseOperation>
{
    using DeviceOp = DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle;

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

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
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

        const index_t Wi            = input_spatial_lengths[0];
        const index_t Wo            = output_spatial_lengths[0];
        const index_t X             = filter_spatial_lengths[0];
        const index_t ConvStrideW   = conv_filter_strides[0];
        const index_t ConvDilationW = conv_filter_dilations[0];
        const index_t InLeftPadW    = input_left_pads[0];
        const index_t InRightPadW   = input_right_pads[0];

        const index_t GemmKTotal = N * Wo;
        const index_t GemmM      = K;
        const index_t GemmN      = C * X;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1Number;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wo, K));

            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wi, C));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wo, K));
            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(X, C)),
                                                       make_merge_transform(make_tuple(N, Wo))),
                                            make_tuple(Sequence<1, 3>{}, Sequence<0, 2>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
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
        const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1Number;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Hi * Wi, C));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

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

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
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

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t Z = filter_spatial_lengths[0];
        const index_t Y = filter_spatial_lengths[1];
        const index_t X = filter_spatial_lengths[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        const index_t ConvDilationD = conv_filter_dilations[0];
        const index_t ConvDilationH = conv_filter_dilations[1];
        const index_t ConvDilationW = conv_filter_dilations[2];

        const index_t InLeftPadD = input_left_pads[0];
        const index_t InLeftPadH = input_left_pads[1];
        const index_t InLeftPadW = input_left_pads[2];

        const index_t InRightPadD = input_right_pads[0];
        const index_t InRightPadH = input_right_pads[1];
        const index_t InRightPadW = input_right_pads[2];

        const index_t GemmKTotal = N * Do * Ho * Wo;
        const index_t GemmM      = K;
        const index_t GemmN      = C * Z * X * Y;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1Number;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K));

            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Di * Hi * Wi, C));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K));
            const auto in_n_di_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
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

            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_dip_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_grid_desc);
        }
    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            1, 1, 1, {1}, {1}, {1}, {1}, {1}, {1}, {1}, 1);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 1);
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
                                                                  1);
    }

    // type convert descs
    template <typename Desc_M0>
    static auto PadDescriptor_M0_1d(Desc_M0 desc_m0, index_t gridSize, index_t blockSize)
    {
        const auto m0           = desc_m0.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * 4;
        const auto pad          = math::integer_least_multiple(m0, loop_step) - m0;
        const auto desc_m0_pad =
            transform_tensor_descriptor(desc_m0,
                                        make_tuple(make_right_pad_transform(m0, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m0_pad;
    }

    template <index_t Dim>
    static auto MakeDescriptor_M0(const std::array<index_t, Dim>& shape,
                                  const std::array<index_t, Dim>& stride,
                                  index_t gridSize,
                                  index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return shape[I]; }, Number<Dim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<Dim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(Dim > 1)
        {
            const auto desc_m0 = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<Dim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M0_1d(desc_m0, gridSize, blockSize);
        }
        else
            return PadDescriptor_M0_1d(desc, gridSize, blockSize);
    }

    using GridDesc_M0 = decltype(MakeDescriptor_M0<1>({1}, {1}, 1, 1));

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
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
                 ck::index_t G,
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
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_G_{G},
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
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
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
                    k_batch_);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            block_2_ctile_map_ =
                GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ =
                N * K *
                std::accumulate(begin(output_spatial_lengths),
                                end(output_spatial_lengths),
                                index_t{1},
                                std::multiplies<>{});
            compute_ptr_offset_of_batch_.BatchStrideB_ =
                N * C *
                std::accumulate(begin(input_spatial_lengths),
                                end(input_spatial_lengths),
                                index_t{1},
                                std::multiplies<>{});
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                K * C *
                std::accumulate(begin(filter_spatial_lengths),
                                end(filter_spatial_lengths),
                                index_t{1},
                                std::multiplies<>{});

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

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        InElementwiseOperation a_element_op_;
        OutElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;

        // for checking IsSupportedArgument()
        index_t Conv_G_;
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> conv_filter_strides_;
        std::array<ck::index_t, NDimSpatial> input_left_pads_;
        std::array<ck::index_t, NDimSpatial> input_right_pads_;
        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
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

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Conv_G_;

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_c_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.c_element_op_,
                                              arg.Conv_G_,
                                              arg.a_grid_desc_kbatch_k0_m_k1_,
                                              arg.b_grid_desc_kbatch_k0_n_k1_,
                                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_ctile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            if(has_main_k0_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
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
        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
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

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
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
                             ck::index_t G,
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
                        G,
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
                        ck::index_t G,
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
                                          G,
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
        str << "DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
