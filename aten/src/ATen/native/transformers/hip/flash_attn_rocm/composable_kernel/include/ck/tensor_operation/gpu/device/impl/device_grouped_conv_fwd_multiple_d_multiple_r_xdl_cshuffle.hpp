// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_multiple_r.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
#include "ck/library/utility/numeric.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <index_t NumDTensor, index_t NumRTensor>
struct ComputePtrOffsetOfStridedBatch
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                   index_t BatchStrideB,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE,
                                   Array<ck::index_t, NumRTensor> BatchStrideRs)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE),
          BatchStrideRs_(BatchStrideRs)
    {
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideA_);
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideB_);
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]); });
        return ds_offset;
    }

    __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    __host__ __device__ constexpr auto GetRsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumRTensor> rs_offset;
        static_for<0, NumRTensor, 1>{}(
            [&](auto i) { rs_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideRs_[i]); });
        return rs_offset;
    }

    index_t BatchStrideA_;
    index_t BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
    Array<ck::index_t, NumRTensor> BatchStrideRs_;
};

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for
 * \link DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the
 * computing of pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename ABDataType,
          typename DsPointer,
          typename EDataType,
          typename RsPointer,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename RsGridDescriptor_MBlock_MPerBlock,
          typename Block2ETileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batch_gemm_multiple_d_xdl_cshuffle(
            const ABDataType* __restrict__ p_a_grid,
            const ABDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            RsPointer p_rs_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const QsElementwiseOperation qs_element_op,
            const RsElementwiseOperation rs_element_op,
            const index_t batch_count,
            const AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
            const RsGridDescriptor_MBlock_MPerBlock rs_grid_desc_mblock_mperblock,
            const Block2ETileMap block_2_ctile_map,
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
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    const auto rs_batch_offset = compute_ptr_offset_of_batch.GetRsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    DsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    RsPointer p_rs_grid_grp;

    static constexpr index_t NumRTensor = RsGridDescriptor_MBlock_MPerBlock::Size();

    static_for<0, NumRTensor, 1>{}(
        [&](auto i) { p_rs_grid_grp(i) = p_rs_grid[i] + rs_batch_offset[i]; });

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_ds_grid_grp,
                                                  p_e_grid + e_batch_offset,
                                                  p_rs_grid_grp,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  qs_element_op,
                                                  rs_element_op,
                                                  a_grid_desc_k0_m_k1,
                                                  b_grid_desc_k0_n_k1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                                  rs_grid_desc_mblock_mperblock,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = p_rs_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = rs_grid_desc_mblock_mperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = qs_element_op;
    ignore = rs_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

} // namespace

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DELayout,
          typename RLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename ReduceAccDataType,
          typename RsDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation,
          typename ThreadReduceOperations,
          typename RsGlobalMemoryDataOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDRThreadTransferClusterLengths_MPerBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          index_t RThreadTransferDstScalarPerVector_MPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGroupedConvFwdMultipleDMultipleR_Xdl_CShuffle
    : public DeviceGroupedConvFwdMultipleDMultipleR<NDimSpatial,
                                                    ALayout,
                                                    BLayout,
                                                    DELayout,
                                                    RLayout,
                                                    ADataType,
                                                    BDataType,
                                                    DsDataType,
                                                    EDataType,
                                                    RsDataType,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    CDEElementwiseOperation,
                                                    RsElementwiseOperation,
                                                    QsElementwiseOperation>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleDMultipleR_Xdl_CShuffle;

    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr index_t NumRTensor = RsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_M_K(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                            const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                            const std::array<index_t, NDimSpatial>& input_left_pads,
                            const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ALay>(a_g_n_c_wis_lengths,
                                                                        a_g_n_c_wis_strides,
                                                                        b_g_k_c_xs_lengths,
                                                                        b_g_k_c_xs_strides,
                                                                        e_g_n_k_wos_lengths,
                                                                        e_g_n_k_wos_strides,
                                                                        conv_filter_strides,
                                                                        conv_filter_dilations,
                                                                        input_left_pads,
                                                                        input_right_pads);

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        return in_gemmm_gemmk_desc;
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_N_K(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>(b_g_k_c_xs_lengths,
                                                                        b_g_k_c_xs_strides);

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        return wei_gemmn_gemmk_desc;
    }

    template <typename ELay>
    static auto
    MakeEGridDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides)
    {
        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<ELay>(e_g_n_k_wos_lengths,
                                                                        e_g_n_k_wos_strides);

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    template <typename Descriptor>
    static auto GetPaddedRGridDescriptor(Descriptor descriptor, index_t MRaw)
    {
        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(
                descriptor,
                make_tuple(make_right_pad_transform(descriptor, MPad)),
                make_tuple(Sequence<0>{}),
                make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return descriptor;
        }
    }

    template <typename RLay,
              typename std::enable_if<is_same_v<RLay, tensor_layout::convolution::GNW> ||
                                          is_same_v<RLay, tensor_layout::convolution::GNHW> ||
                                          is_same_v<RLay, tensor_layout::convolution::GNDHW>,
                                      bool>::type = false>
    static auto
    MakeRGridDescriptor_M(const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
                          const std::array<index_t, NDimSpatial + 2>& /* r_g_n_wos_strides */)
    {
        const index_t N = r_g_n_wos_lengths[1];

        const index_t NHoWo =
            N * ck::accumulate_n<index_t>(
                    r_g_n_wos_lengths.begin() + 2, NDimSpatial, 1, std::multiplies<>());

        const auto r_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(NHoWo));

        return GetPaddedRGridDescriptor(r_grid_desc_mraw, NHoWo);
    }

    template <typename RLay,
              typename std::enable_if<is_same_v<RLay, tensor_layout::convolution::G_NW> ||
                                          is_same_v<RLay, tensor_layout::convolution::G_NHW> ||
                                          is_same_v<RLay, tensor_layout::convolution::G_NDHW> ||
                                          is_same_v<RLay, tensor_layout::convolution::NWG> ||
                                          is_same_v<RLay, tensor_layout::convolution::NHWG> ||
                                          is_same_v<RLay, tensor_layout::convolution::NDHWG>,
                                      bool>::type = false>
    static auto MakeRGridDescriptor_M(const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
                                      const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_strides)
    {
        const index_t N = r_g_n_wos_lengths[1];

        const index_t WoStride = r_g_n_wos_strides[NDimSpatial + 2];

        const index_t NHoWo =
            N * ck::accumulate_n<index_t>(
                    r_g_n_wos_lengths.begin() + 2, NDimSpatial, 1, std::multiplies<>());

        const auto r_grid_desc_mraw =
            make_naive_tensor_descriptor(make_tuple(NHoWo), make_tuple(WoStride));

        return GetPaddedRGridDescriptor(r_grid_desc_mraw, NHoWo);
    }

    using AGridDesc_M_K = remove_cvref_t<decltype(
        MakeAGridDescriptor_M_K<ALayout>({}, {}, {}, {}, {}, {}, {}, {}, {}, {}))>;
    using BGridDesc_N_K = remove_cvref_t<decltype(MakeBGridDescriptor_N_K<BLayout>({}, {}))>;
    using EGridDesc_M_N = remove_cvref_t<decltype(MakeEGridDescriptor_M_N<DELayout>({}, {}))>;
    using RGridDesc_M   = remove_cvref_t<decltype(MakeRGridDescriptor_M<RLayout>({}, {}))>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleDMultipleR_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        ReduceAccDataType,
        RsDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        QsElementwiseOperation,
        RsElementwiseOperation,
        ThreadReduceOperations,
        InMemoryDataOperationEnum::Set,
        RsGlobalMemoryDataOperation,
        AGridDesc_M_K,
        BGridDesc_N_K,
        EGridDesc_M_N,
        RGridDesc_M,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDRThreadTransferClusterLengths_MPerBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        RThreadTransferDstScalarPerVector_MPerBlock,
        LoopSched>;

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;

    using Block2ETileMap = typename GridwiseGemm::DefaultBlock2ETileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,
                 const void* p_b,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 std::array<void*, NumRTensor> p_rs,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
                 const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op,
                 const QsElementwiseOperation& qs_element_op,
                 const RsElementwiseOperation& rs_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a)},
              p_b_grid_{static_cast<const BDataType*>(p_b)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              p_rs_grid_{}, // FIXME
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K<ALayout>(a_g_n_c_wis_lengths,
                                                                          a_g_n_c_wis_strides,
                                                                          b_g_k_c_xs_lengths,
                                                                          b_g_k_c_xs_strides,
                                                                          e_g_n_k_wos_lengths,
                                                                          e_g_n_k_wos_strides,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K<BLayout>(b_g_k_c_xs_lengths,
                                                                          b_g_k_c_xs_strides)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<DELayout>(e_g_n_k_wos_lengths,
                                                                           e_g_n_k_wos_strides)},
              r_grid_desc_m_{
                  DeviceOp::MakeRGridDescriptor_M<RLayout>(r_g_n_wos_lengths, r_g_n_wos_strides)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              rs_grid_desc_mblock_mperblock_{},
              block_2_etile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              compute_ptr_offset_of_batch_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              qs_element_op_{qs_element_op},
              rs_element_op_{rs_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // A/B/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_k_wos_strides[0];

            // populate desc for Ds/E
            if(GridwiseGemm::CheckValidity(a_grid_desc_m_k_,
                                           b_grid_desc_n_k_,
                                           e_grid_desc_m_n_,
                                           r_grid_desc_m_,
                                           block_2_etile_map_))
            {
                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);

                // populate pointer, batch stride, desc for Ds
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                    // D pointer
                    p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);

                    // D batch stride
                    compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_k_wos_strides[i][0];

                    // D desc
                    ds_grid_desc_m_n_(i) = DeviceOp::MakeEGridDescriptor_M_N<DELayout>(
                        ds_g_n_k_wos_lengths[i], ds_g_n_k_wos_strides[i]);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_(i) =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            ds_grid_desc_m_n_(i));
                });

                // populate pointer for Rs
                static_for<0, NumRTensor, 1>{}([&](auto i) {
                    using RDataType = remove_cvref_t<tuple_element_t<i.value, RsDataType>>;

                    // R pointer
                    p_rs_grid_(i) = static_cast<RDataType*>(p_rs[i]);

                    rs_grid_desc_mblock_mperblock_(i) =
                        GridwiseGemm::MakeRGridDescriptor_MBlock_MPerBlock(r_grid_desc_m_);
                });
            }
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;
        typename GridwiseGemm::RsGridPointer p_rs_grid_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        EGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;
        RGridDesc_M r_grid_desc_m_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        StaticallyIndexedArray<
            typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            NumDTensor>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_; // FIXME: Ds desc may be of different
                                                             // type from E
        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;

        StaticallyIndexedArray<typename GridwiseGemm::RGridDescriptor_MBlock_MPerBlock, NumRTensor>
            rs_grid_desc_mblock_mperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        ComputePtrOffsetOfStridedBatch<NumDTensor, NumRTensor> compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        QsElementwiseOperation qs_element_op_;
        RsElementwiseOperation rs_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            arg.e_grid_desc_m_n_,
                                            arg.r_grid_desc_m_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmMultipleD_xdl_cshuffle has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) *
                arg.a_g_n_c_wis_lengths_[0]; // Group count

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_batch_gemm_multiple_d_xdl_cshuffle<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    typename GridwiseGemm::RsGridPointer,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    QsElementwiseOperation,
                    RsElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    ck::StaticallyIndexedArray<
                        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        NumDTensor>,
                    typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    ck::StaticallyIndexedArray<
                        typename GridwiseGemm::RGridDescriptor_MBlock_MPerBlock,
                        NumRTensor>,
                    Block2ETileMap,
                    ComputePtrOffsetOfStridedBatch<NumDTensor, NumRTensor>,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.p_rs_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.qs_element_op_,
                                              arg.rs_element_op_,
                                              arg.a_g_n_c_wis_lengths_[0], // Group count
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.rs_grid_desc_mblock_mperblock_,
                                              arg.block_2_etile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
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

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        // check device
        if(get_device_name() == "gfx908")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else if(get_device_name() == "gfx90a")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t> || is_same_v<AccDataType, double>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 2];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X        = arg.b_g_k_c_xs_lengths_[i + 2];
                const index_t LeftPad  = arg.input_left_pads_[i];
                const index_t RightPad = arg.input_right_pads_[i];

                if(!(X == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC>)
        {
            const index_t C = arg.a_g_n_c_wis_lengths_[2];

            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::G_K_X_C> || is_same_v<BLayout, ctc::G_K_YX_C> ||
                     is_same_v<BLayout, ctc::G_K_ZYX_C> || is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC> ||
                     is_same_v<BLayout, ctc::KXGC> || is_same_v<BLayout, ctc::KYXGC> ||
                     is_same_v<BLayout, ctc::KZYXGC>)

        {
            const index_t C = arg.b_g_k_c_xs_lengths_[2];

            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        //  check vector access of Ds
        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            // FIXME: layout
            if constexpr(is_same_v<DELayout, ctc::G_NW_K> || is_same_v<DELayout, ctc::G_NHW_K> ||
                         is_same_v<DELayout, ctc::G_NDHW_K> || is_same_v<DELayout, ctc::GNWK> ||
                         is_same_v<DELayout, ctc::GNHWK> || is_same_v<DELayout, ctc::GNDHWK> ||
                         is_same_v<DELayout, ctc::NWGK> || is_same_v<DELayout, ctc::NHWGK> ||
                         is_same_v<DELayout, ctc::NDHWGK>)
            {
                const index_t K = arg.ds_g_n_k_wos_lengths_[i][2];

                if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    valid = false;
                }
            }
            else
            {
                valid = false;
            }
        });

        if(!valid)
        {
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<DELayout, ctc::G_NW_K> || is_same_v<DELayout, ctc::G_NHW_K> ||
                     is_same_v<DELayout, ctc::G_NDHW_K> || is_same_v<DELayout, ctc::GNWK> ||
                     is_same_v<DELayout, ctc::GNHWK> || is_same_v<DELayout, ctc::GNDHWK> ||
                     is_same_v<DELayout, ctc::NWGK> || is_same_v<DELayout, ctc::NHWGK> ||
                     is_same_v<DELayout, ctc::NDHWGK>)
        {
            const index_t K = arg.e_g_n_k_wos_lengths_[2];

            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of R
        if constexpr(!(is_same_v<RLayout, ctc::G_NW> || is_same_v<RLayout, ctc::G_NHW> ||
                       is_same_v<RLayout, ctc::G_NDHW> || is_same_v<RLayout, ctc::GNW> ||
                       is_same_v<RLayout, ctc::GNHW> || is_same_v<RLayout, ctc::GNDHW> ||
                       is_same_v<RLayout, ctc::NWG> || is_same_v<RLayout, ctc::NHWG> ||
                       is_same_v<RLayout, ctc::NDHWG>))
        {
            return false;
        }

        // check Gridwise GEMM
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                           arg.b_grid_desc_n_k_,
                                           arg.e_grid_desc_m_n_,
                                           arg.r_grid_desc_m_,
                                           arg.block_2_etile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        std::array<void*, NumRTensor> p_rs,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op,
        const QsElementwiseOperation& qs_element_op,
        const RsElementwiseOperation& rs_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        p_rs,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        r_g_n_wos_lengths,
                        r_g_n_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        qs_element_op,
                        rs_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        std::array<void*, NumRTensor> p_rs,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op,
        const QsElementwiseOperation& qs_element_op,
        const RsElementwiseOperation& rs_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          p_rs,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          r_g_n_wos_lengths,
                                          r_g_n_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
                                          qs_element_op,
                                          rs_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
