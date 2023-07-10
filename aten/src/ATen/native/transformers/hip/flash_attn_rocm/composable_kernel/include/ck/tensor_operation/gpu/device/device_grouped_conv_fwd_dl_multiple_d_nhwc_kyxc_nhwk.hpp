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
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                   index_t BatchStrideB,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
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

    index_t BatchStrideA_;
    index_t BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
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
 * device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for \link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
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
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_K0_M0_M1_K1,
          typename BGridDesc_K0_N0_N1_K1,
          typename DsGridDesc_M0_M10_M11_N0_N10_N11,
          typename CGridDesc_M0_M10_M11_N0_N10_N11,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_conv_fwd_dl_multiple_d(
            const ABDataType* __restrict__ p_a_grid,
            const ABDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const index_t batch_count,
            const AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1,
            const BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1,
            const DsGridDesc_M0_M10_M11_N0_N10_N11 ds_grid_desc_m0_m10_m11_n0_n10_n11,
            const CGridDesc_M0_M10_M11_N0_N10_N11 e_grid_desc_m0_m10_m11_n0_n10_n11,
            const Block2CTileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__) || defined(__gfx1030__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(ABDataType);

    __shared__ ABDataType p_shared[shared_block_size];

    DsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor = DsGridDesc_M0_M10_M11_N0_N10_N11::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::Run(p_a_grid + a_batch_offset,
                      p_b_grid + b_batch_offset,
                      p_ds_grid_grp,
                      p_e_grid + c_batch_offset,
                      p_shared,
                      a_element_op,
                      b_element_op,
                      cde_element_op,
                      a_grid_desc_k0_m0_m1_k1,
                      b_grid_desc_k0_n0_n1_k1,
                      ds_grid_desc_m0_m10_m11_n0_n10_n11,
                      e_grid_desc_m0_m10_m11_n0_n10_n11,
                      block_2_ctile_map,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m0_m1_k1;
    ignore = b_grid_desc_k0_n0_n1_k1;
    ignore = ds_grid_desc_m0_m10_m11_n0_n10_n11;
    ignore = e_grid_desc_m0_m10_m11_n0_n10_n11;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;

    compute_ptr_offset_of_batch.GetAPtrOffset(0);
    compute_ptr_offset_of_batch.GetBPtrOffset(0);
    compute_ptr_offset_of_batch.GetEPtrOffset(0);
#endif
}
} // namespace

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Forward convolution with up to 3 spatial dimentions
//  @li         Input tensor in GNWC data format
//  @li         Weight tensor in GKXC data format
//  @li         Output tensor in GNWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <index_t NDimSpatial,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1,
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
struct DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
    : public DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                           ALayout,
                                           BLayout,
                                           DsLayout,
                                           ELayout,
                                           ADataType,
                                           BDataType,
                                           DsDataType,
                                           EDataType,
                                           AElementwiseOperation,
                                           BElementwiseOperation,
                                           CDEElementwiseOperation>
{
    using DeviceOp = DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, K0PerBlock};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_AK0_M_AK1(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
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

        const auto M = in_gemmm_gemmk_desc.GetLength(I0);
        const auto K = in_gemmm_gemmk_desc.GetLength(I1);

        const auto AK0 = K / K1;

        return transform_tensor_descriptor(
            in_gemmm_gemmk_desc,
            make_tuple(make_unmerge_transform(make_tuple(AK0, K1)), make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_BK0_N_BK1(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                                  const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>(b_g_k_c_xs_lengths,
                                                                        b_g_k_c_xs_strides);

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        const auto N = wei_gemmn_gemmk_desc.GetLength(I0);
        const auto K = wei_gemmn_gemmk_desc.GetLength(I1);

        const auto BK0 = K / K1;

        return transform_tensor_descriptor(
            wei_gemmn_gemmk_desc,
            make_tuple(make_unmerge_transform(make_tuple(BK0, K1)), make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
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

    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(ds_g_n_k_wos_lengths[i],
                                                                  ds_g_n_k_wos_strides[i]);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        MakeAGridDescriptor_AK0_M_AK1<ALayout>({}, {}, {}, {}, {}, {}, {}, {}, {}, {}))>;
    using BGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(MakeBGridDescriptor_BK0_N_BK1<BLayout>({}, {}))>;
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}))>;
    using EGridDesc_M_N  = remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>({}, {}))>;

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDlMultipleD_km_kn_mn<BlockSize,
                                         ADataType,
                                         AccDataType,
                                         DsDataType,
                                         EDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation,
                                         InMemoryDataOperationEnum::Set,
                                         AGridDesc_AK0_M_AK1,
                                         BGridDesc_BK0_N_BK1,
                                         EGridDesc_M_N,
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
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_AK0_M_AK1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_BK0_N_BK1{}));
    using DsGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(DsGridDesc_M_N{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(EGridDesc_M_N{}));
    using DefaultBlock2CTileMap =
        decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(EGridDesc_M_N{}));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,
                 const void* p_b,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
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
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a)},
              p_b_grid_{static_cast<const BDataType*>(p_b)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{a_g_n_c_wis_lengths[0]},
              a_grid_desc_ak0_m_ak1_{
                  DeviceOp::MakeAGridDescriptor_AK0_M_AK1<ALayout>(a_g_n_c_wis_lengths,
                                                                   a_g_n_c_wis_strides,
                                                                   b_g_k_c_xs_lengths,
                                                                   b_g_k_c_xs_strides,
                                                                   e_g_n_k_wos_lengths,
                                                                   e_g_n_k_wos_strides,
                                                                   conv_filter_strides,
                                                                   conv_filter_dilations,
                                                                   input_left_pads,
                                                                   input_right_pads)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1<BLayout>(
                  b_g_k_c_xs_lengths, b_g_k_c_xs_strides)},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<ELayout>(e_g_n_k_wos_lengths,
                                                                          e_g_n_k_wos_strides)},
              a_grid_desc_k0_m0_m1_k1_{},
              b_grid_desc_k0_n0_n1_k1_{},
              ds_grid_desc_m0_m10_m11_n0_n10_n11_{},
              e_grid_desc_m0_m10_m11_n0_n10_n11_{},
              block_2_ctile_map_{},
              compute_ptr_offset_of_batch_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
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

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);

                // D batch stride
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_k_wos_strides[i][0];

                // D desc
                ds_grid_desc_m_n_(i) = DeviceOp::MakeEGridDescriptor_M_N<DLayout>(
                    ds_g_n_k_wos_lengths[i], ds_g_n_k_wos_strides[i]);
            });

            // populate desc for Ds/E
            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_ak0_m_ak1_, b_grid_desc_bk0_n_bk1_, e_grid_desc_m_n_))
            {

                a_grid_desc_k0_m0_m1_k1_ =
                    GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_ak0_m_ak1_);
                b_grid_desc_k0_n0_n1_k1_ =
                    GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_bk0_n_bk1_);
                e_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(e_grid_desc_m_n_);

                ds_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(ds_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(e_grid_desc_m_n_);
            }
        }

        void Print() const
        {
            std::cout << "A[K0, M, K1]: " << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "B[K0, N, K1]: " << b_grid_desc_bk0_n_bk1_ << std::endl;
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
            std::cout << "num_group: " << num_group_ << std::endl;

            std::cout << "A[k0, m0, m1, k1]: " << a_grid_desc_k0_m0_m1_k1_ << std::endl;
            std::cout << "B[k0, n0, n1, k1]: " << b_grid_desc_k0_n0_n1_k1_ << std::endl;
            std::cout << "A[m0, m10, m11, n0, n10, n11]: " << e_grid_desc_m0_m10_m11_n0_n10_n11_
                      << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        index_t num_group_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1_;
        BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1_;
        DsGridDesc_M0_M10_M11_N0_N10_N11 ds_grid_desc_m0_m10_m11_n0_n10_n11_;
        CGridDesc_M0_M10_M11_N0_N10_N11 e_grid_desc_m0_m10_m11_n0_n10_n11_;

        // block-to-e-tile map
        DefaultBlock2CTileMap block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<NumDTensor> compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

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

        float Run(const Argument& arg, const StreamConfig& stream_config)
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.e_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK has invalid setting");
            }

            const index_t grid_size =
                GridwiseGemm::CalculateGridSize(arg.e_grid_desc_m_n_.GetLength(I0),
                                                arg.e_grid_desc_m_n_.GetLength(I1)) *
                arg.num_group_;

            auto launch_kernel = [&](auto has_main_k_block_loop,
                                     auto has_double_tail_k_block_loop) {
                constexpr bool has_main_loop   = has_main_k_block_loop.value;
                constexpr bool has_double_loop = has_double_tail_k_block_loop;

                const auto kernel = kernel_grouped_conv_fwd_dl_multiple_d<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AGridDesc_K0_M0_M1_K1,
                    DeviceOp::BGridDesc_K0_N0_N1_K1,
                    DeviceOp::DsGridDesc_M0_M10_M11_N0_N10_N11,
                    DeviceOp::CGridDesc_M0_M10_M11_N0_N10_N11,
                    DefaultBlock2CTileMap,
                    ComputePtrOffsetOfStridedBatch<NumDTensor>,
                    has_main_loop,
                    has_double_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_g_n_c_wis_lengths_[0], // Group count
                                              arg.a_grid_desc_k0_m0_m1_k1_,
                                              arg.b_grid_desc_k0_n0_n1_k1_,
                                              arg.ds_grid_desc_m0_m10_m11_n0_n10_n11_,
                                              arg.e_grid_desc_m0_m10_m11_n0_n10_n11_,
                                              arg.block_2_ctile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            const auto K0                    = arg.a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, true>{});
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, false>{});
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, false>{});
            }
            return 0;
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
        if(!(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030"))
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
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    std::cout << "Filter1x1Stride1Pad0 check: XY_index = " << i << " X = " << X
                              << " ConvStride = " << ConvStride << " LeftPad = " << LeftPad
                              << " RightPad = " << RightPad << std::endl;
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
                const index_t X        = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad  = arg.input_left_pads_[i];
                const index_t RightPad = arg.input_right_pads_[i];

                if(!(X == 1 && LeftPad == 0 && RightPad == 0))
                {
                    std::cout << "Filter1x1Stride1Pad0 check: XY_index = " << i << " X = " << X
                              << " LeftPad = " << LeftPad << " RightPad = " << RightPad
                              << std::endl;
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
            auto srcVectorLengths = ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1{};
            if(srcVectorLengths[I1] != 1 || srcVectorLengths[I2] != 1)
            {
                return false;
            }
            if(K1 % srcVectorLengths[I3] != 0 || K0PerBlock % srcVectorLengths[I0] != 0)
            {
                return false;
            }

            const index_t C = arg.a_g_n_c_wis_lengths_[2];

            if(C % (srcVectorLengths[I0] * srcVectorLengths[I3]) != 0)
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
            auto srcVectorLengths = BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1{};
            if(srcVectorLengths[I1] != 1 || srcVectorLengths[I2] != 1)
            {
                return false;
            }
            if(K1 % srcVectorLengths[I3] != 0 || K0PerBlock % srcVectorLengths[I0] != 0)
            {
                return false;
            }

            const index_t C = arg.b_g_k_c_xs_lengths_[2];

            if(C % (srcVectorLengths[I0] * srcVectorLengths[I3]) != 0)
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK>)
        {
            const index_t K = arg.e_g_n_k_wos_lengths_[2];

            if(!(K % CThreadTransferDstScalarPerVector == 0 && CThreadTransferSrcDstVectorDim == 5))
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        // check Gridwise GEMM
        return GridwiseGemm::CheckValidity(
            arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.e_grid_desc_m_n_);
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
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
