// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_GRIDWISE_CONTRACTION_DLOPS_V1R2_HPP
#define CK_GRIDWISE_CONTRACTION_DLOPS_V1R2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_dlops_v2r3.hpp"
#include "blockwise_tensor_slice_transfer_v2.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_set.hpp"

namespace ck {

template <typename GridwiseContraction,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_GK0_GM0_GM10_GM11_GK1,
          typename BGridDesc_GK0_GN0_GN10_GN11_GK1,
          typename CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1,
          typename CGridBlockCluster_BlockId_To_GM10_GN10,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_contraction_dlops_v1r2(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_GK0_GM0_GM10_GM11_GK1 a_grid_desc_gk0_gm0_gm10_gm11_gk1,
            const BGridDesc_GK0_GN0_GN10_GN11_GK1 b_grid_desc_gk0_gn0_gn10_gn11_gk1,
            const CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1 c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
            const CGridBlockCluster_BlockId_To_GM10_GN10 c_grid_block_cluster_blockid_to_gm10_gn10)
{
    constexpr index_t shared_block_size =
        GridwiseContraction::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseContraction::Run(p_a_grid,
                             p_b_grid,
                             p_c_grid,
                             p_shared_block,
                             a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                             b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                             c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
                             c_grid_block_cluster_blockid_to_gm10_gn10,
                             integral_constant<bool, HasMainKBlockLoop>{},
                             integral_constant<bool, HasDoubleTailKBlockLoop>{});
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_GK0_GM0_GM1_GK1,
          typename BGridDesc_GK0_GN0_GN1_GK1,
          typename CGridDesc_GM0_GM1_GN0_GN1,
          index_t GM1PerBlockGM11,
          index_t GN1PerBlockGN11,
          index_t GK0PerBlock,
          index_t BM1PerThreadBM11,
          index_t BN1PerThreadBN11,
          index_t BK0PerThread,
          typename BM10BN10ThreadClusterBM10Xs,
          typename BM10BN10ThreadClusterBN10Xs,
          typename ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridStepHacks,
          typename BGridStepHacks,
          typename CGridStepHacks,
          typename AGridMoveSliceWindowStepHacks,
          typename BGridMoveSliceWindowStepHacks>
struct GridwiseContractionDlops_A_GK0_GM0_GM1_GK1_B_GK0_GN0_GN1_GK1_C_GM0_GM1_GN0_GN1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // GM0 and GN0 need to known at compile-time
    static constexpr auto GM0 = CGridDesc_GM0_GM1_GN0_GN1{}.GetLength(I0);
    static constexpr auto GN0 = CGridDesc_GM0_GM1_GN0_GN1{}.GetLength(I2);
    static constexpr auto GK1 = AGridDesc_GK0_GM0_GM1_GK1{}.GetLength(I3);

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // lds max alignment
        // TODO: part of them should be moved into blockwise-gemm
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = GK1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_block_desc_gk0_gm0_gm10_gm11_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GM0, I1, Number<GM1PerBlockGM11>{}, GK1),
            max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_block_desc_gk0_gn0_gn10_gn11_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GN0, I1, Number<GN1PerBlockGN11>{}, GK1),
            max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_block_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size = math::integer_least_multiple(
            b_block_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_aligned_space_size + b_block_aligned_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_GK0_GM0_GM1_GK1& a_grid_desc_gk0_gm0_gm1_gk1,
                  const BGridDesc_GK0_GN0_GN1_GK1& b_grid_desc_gk0_gn0_gn1_gk1,
                  const CGridDesc_GM0_GM1_GN0_GN1& c_grid_desc_gm0_gm1_gn0_gn1)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(GM0)>>::value &&
                          is_known_at_compile_time<remove_cv_t<decltype(GN0)>>::value,
                      "wrong! GM0 and GN0 need to be known at compile-time");

        const auto GM1 = a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I2);
        const auto GN1 = b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I2);
        const auto GK0 = a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I0);

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        return (
            (GM0 == c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I0) &&
             GM1 == c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I1) &&
             GN0 == c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I2) &&
             GN1 == c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I3) &&
             GM0 == a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I1) &&
             GM1 == a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I2) &&
             GN0 == b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I1) &&
             GN1 == b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I2) &&
             GK0 == b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I0) &&
             GK1 == b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I3)) &&
            (GM1 % GM1PerBlockGM11 == 0 && GN1 % GN1PerBlockGN11 == 0 && GK0 % GK0PerBlock == 0));
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_GM0_GM1_GN0_GN1& c_grid_desc_gm0_gm1_gn0_gn1)
    {
        const auto GM1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I1);
        const auto GN1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I3);

        constexpr index_t GM11 = GM1PerBlockGM11;
        constexpr index_t GN11 = GN1PerBlockGN11;

        const index_t GM10 = GM1 / GM11;
        const index_t GN10 = GN1 / GN11;

        const index_t grid_size = GM10 * GN10;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t GK0)
    {
        const bool has_main_k_block_loop = (GK0 + GK0PerBlock) / (2 * GK0PerBlock) > 1;

        return has_main_k_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailKBlockLoop(index_t GK0)
    {
        const bool has_double_tail_k_block_loop = (GK0 / GK0PerBlock) % 2 == 0;

        return has_double_tail_k_block_loop;
    }

    __host__ __device__ static constexpr auto MakeAGridDescriptor_GK0_GM0_GM10_GM11_GK1(
        const AGridDesc_GK0_GM0_GM1_GK1& a_grid_desc_gk0_gm0_gm1_gk1)
    {
        const auto GK0 = a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I0);
        const auto GM1 = a_grid_desc_gk0_gm0_gm1_gk1.GetLength(I2);

        const auto GM11 = Number<GM1PerBlockGM11>{};
        const auto GM10 = GM1 / GM11;

        const auto a_grid_desc_gk0_gm0_gm10_gm11_gk1 = transform_tensor_descriptor(
            a_grid_desc_gk0_gm0_gm1_gk1,
            make_tuple(make_pass_through_transform(GK0),
                       make_pass_through_transform(GM0),
                       make_unmerge_transform(make_tuple(GM10, GM11)),
                       make_pass_through_transform(GK1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return a_grid_desc_gk0_gm0_gm10_gm11_gk1;
    }

    __host__ __device__ static constexpr auto MakeBGridDescriptor_GK0_GN0_GN10_GN11_GK1(
        const BGridDesc_GK0_GN0_GN1_GK1& b_grid_desc_gk0_gn0_gn1_gk1)
    {
        const auto GK0 = b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I0);
        const auto GN1 = b_grid_desc_gk0_gn0_gn1_gk1.GetLength(I2);

        const auto GN11 = Number<GN1PerBlockGN11>{};
        const auto GN10 = GN1 / GN11;

        const auto b_grid_desc_gk0_gn0_gn10_gn11_gk1 = transform_tensor_descriptor(
            b_grid_desc_gk0_gn0_gn1_gk1,
            make_tuple(make_pass_through_transform(GK0),
                       make_pass_through_transform(GN0),
                       make_unmerge_transform(make_tuple(GN10, GN11)),
                       make_pass_through_transform(GK1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return b_grid_desc_gk0_gn0_gn10_gn11_gk1;
    }

    __host__ __device__ static constexpr auto MakeCGridDescriptor_GM10_BM0_BM1_GN10_BN0_BN1(
        const CGridDesc_GM0_GM1_GN0_GN1& c_grid_desc_gm0_gm1_gn0_gn1)
    {
        const auto GM1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I1);
        const auto GN1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I3);

        constexpr auto GM11 = Number<GM1PerBlockGM11>{};
        constexpr auto GN11 = Number<GN1PerBlockGN11>{};

        const auto GM10 = GM1 / GM11;
        const auto GN10 = GN1 / GN11;

        constexpr auto BM = GM0 * GM11;
        constexpr auto BN = GN0 * GN11;

        constexpr auto BM1 =
            Number<container_reduce(BM10BN10ThreadClusterBM10Xs{}, math::multiplies{}, I1) *
                   BM1PerThreadBM11>{};
        constexpr auto BN1 =
            Number<container_reduce(BM10BN10ThreadClusterBN10Xs{}, math::multiplies{}, I1) *
                   BN1PerThreadBN11>{};

        constexpr auto BM0 = BM / BM1;
        constexpr auto BN0 = BN / BN1;

        const auto c_gm0_gm10_gm11_gn0_gn10_gn11_grid_desc = transform_tensor_descriptor(
            c_grid_desc_gm0_gm1_gn0_gn1,
            make_tuple(make_pass_through_transform(GM0),
                       make_unmerge_transform(make_tuple(GM10, GM11)),
                       make_pass_through_transform(GN0),
                       make_unmerge_transform(make_tuple(GN10, GN11))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}));

        const auto c_gm10_bm_gn10_bn_grid_desc = transform_tensor_descriptor(
            c_gm0_gm10_gm11_gn0_gn10_gn11_grid_desc,
            make_tuple(make_pass_through_transform(GM10),
                       make_merge_transform(make_tuple(GM0, GM11)),
                       make_pass_through_transform(GN10),
                       make_merge_transform(make_tuple(GN0, GN11))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}, Sequence<4>{}, Sequence<3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1 = transform_tensor_descriptor(
            c_gm10_bm_gn10_bn_grid_desc,
            make_tuple(make_pass_through_transform(GM10),
                       make_unmerge_transform(make_tuple(BM0, BM1)),
                       make_pass_through_transform(GN10),
                       make_unmerge_transform(make_tuple(BN0, BN1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}));

        return c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1;
    }

    __host__ __device__ static constexpr auto MakeCGridBlockCluster_BlockId_To_GM10_GN10(
        const CGridDesc_GM0_GM1_GN0_GN1& c_grid_desc_gm0_gm1_gn0_gn1)
    {
        const auto GM1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I1);
        const auto GN1 = c_grid_desc_gm0_gm1_gn0_gn1.GetLength(I3);

        constexpr auto GM11 = Number<GM1PerBlockGM11>{};
        constexpr auto GN11 = Number<GN1PerBlockGN11>{};

        const auto GM10 = GM1 / GM11;
        const auto GN10 = GN1 / GN11;

        const auto c_grid_block_cluster_blockid_to_gm10_gn10 = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(GM10, GN10))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return c_grid_block_cluster_blockid_to_gm10_gn10;
    }

    using AGridDesc_GK0_GM0_GM10_GM11_GK1 =
        decltype(MakeAGridDescriptor_GK0_GM0_GM10_GM11_GK1(AGridDesc_GK0_GM0_GM1_GK1{}));
    using BGridDesc_GK0_GN0_GN10_GN11_GK1 =
        decltype(MakeBGridDescriptor_GK0_GN0_GN10_GN11_GK1(BGridDesc_GK0_GN0_GN1_GK1{}));
    using CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1 =
        decltype(MakeCGridDescriptor_GM10_BM0_BM1_GN10_BN0_BN1(CGridDesc_GM0_GM1_GN0_GN1{}));
    using CGridBlockCluster_BlockId_To_GM10_GN10 =
        decltype(MakeCGridBlockCluster_BlockId_To_GM10_GN10(CGridDesc_GM0_GM1_GN0_GN1{}));

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        FloatAB* __restrict__ p_shared_block,
        const AGridDesc_GK0_GM0_GM10_GM11_GK1& a_grid_desc_gk0_gm0_gm10_gm11_gk1,
        const BGridDesc_GK0_GN0_GN10_GN11_GK1& b_grid_desc_gk0_gn0_gn10_gn11_gk1,
        const CGridDesc_GM10_BM0_BM1_GN10_BN0_BN1& c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
        const CGridBlockCluster_BlockId_To_GM10_GN10& c_grid_block_cluster_blockid_to_gm10_gn10,
        integral_constant<bool, HasMainKBlockLoop>,
        integral_constant<bool, HasDoubleTailKBlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1.GetElementSpaceSize());

        const auto GK0 = a_grid_desc_gk0_gm0_gm10_gm11_gk1.GetLength(I0);

        // divide block work by [GM10, GN10]
        const auto c_gm10_gn10_block_cluster_idx =
            c_grid_block_cluster_blockid_to_gm10_gn10.CalculateBottomIndex(
                make_multi_index(get_block_1d_id()));

        // HACK: this force index data into SGPR
        const index_t igm10 = __builtin_amdgcn_readfirstlane(c_gm10_gn10_block_cluster_idx[I0]);
        const index_t ign10 = __builtin_amdgcn_readfirstlane(c_gm10_gn10_block_cluster_idx[I1]);

        // lds max alignment
        // TODO: part of them should be moved into blockwise-gemm
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = GK1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_block_desc_gk0_gm0_gm10_gm11_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GM0, I1, Number<GM1PerBlockGM11>{}, GK1),
            max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_block_desc_gk0_gn0_gn10_gn11_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GN0, I1, Number<GN1PerBlockGN11>{}, GK1),
            max_lds_align);

        // A matrix in LDS memory for blockwise GEMM
        //   be careful of LDS alignment
        constexpr auto a_block_desc_gk0_bm_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GM0 * Number<GM1PerBlockGM11>{}, GK1), max_lds_align);

        // B matrix in LDS memory for blockwise GEMM
        //   be careful of LDS alignment
        constexpr auto b_block_desc_gk0_bn_gk1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<GK0PerBlock>{}, GN0 * Number<GN1PerBlockGN11>{}, GK1), max_lds_align);

        static_assert(a_block_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize() ==
                              a_block_desc_gk0_bm_gk1.GetElementSpaceSize() &&
                          b_block_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize() ==
                              b_block_desc_gk0_bn_gk1.GetElementSpaceSize(),
                      "wrong!");

        // A matrix blockwise copy
        auto a_blockwise_copy = BlockwiseTensorSliceTransfer_v5r1<
            BlockSize,
            InMemoryDataOperationEnum::Set,
            Sequence<GK0PerBlock, GM0, 1, GM1PerBlockGM11, GK1.value>,
            ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(a_grid_desc_gk0_gm0_gm10_gm11_gk1),
            decltype(a_block_desc_gk0_gm0_gm10_gm11_gk1),
            ABlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3, 4>,
            ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1, // SrcVectorTensorLengths
            ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1, // DstVectorTensorLengths
            ABlockTransferSrcVectorTensorContiguousDimOrder, // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3, 4>,                         // DstVectorTensorContiguousDimOrder
            false,
            true>(a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                  make_multi_index(0, 0, igm10, 0, 0),
                  a_block_desc_gk0_gm0_gm10_gm11_gk1,
                  make_multi_index(0, 0, 0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy = BlockwiseTensorSliceTransfer_v5r1<
            BlockSize,
            InMemoryDataOperationEnum::Set,
            Sequence<GK0PerBlock, GN0, 1, GN1PerBlockGN11, GK1.value>,
            BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(b_grid_desc_gk0_gn0_gn10_gn11_gk1),
            decltype(b_block_desc_gk0_gn0_gn10_gn11_gk1),
            BBlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3, 4>,
            BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1, // SrcVectorTensorLengths
            BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1, // DstVectorTensorLengths
            BBlockTransferSrcVectorTensorContiguousDimOrder, // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3, 4>,                         // DstVectorTensorContiguousDimOrder
            false,
            true>(b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                  make_multi_index(0, 0, ign10, 0, 0),
                  b_block_desc_gk0_gn0_gn10_gn11_gk1,
                  make_multi_index(0, 0, 0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[GK0PerBlock, GM1PerBlockGM11] is in LDS
        //     b_mtx[KPerBlocl, GN1PerBlockGN11] is in LDS
        //     c_mtx[GM1PerBlockGM11, GN1PerBlockGN11] is distributed among threads, and saved in
        //       register
        const auto blockwise_gemm =
            BlockwiseGemmDlops_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2<
                BlockSize,
                FloatAB,
                FloatAB,
                FloatAcc,
                decltype(a_block_desc_gk0_bm_gk1),
                decltype(b_block_desc_gk0_bn_gk1),
                BM1PerThreadBM11,
                BN1PerThreadBN11,
                BK0PerThread,
                BM10BN10ThreadClusterBM10Xs,
                BM10BN10ThreadClusterBN10Xs,
                BM1PerThreadBM11,
                BN1PerThreadBN11>{};

        constexpr auto c_thread_tensor_lengths_bm0_bm1_bn0_bn1 =
            decltype(blockwise_gemm)::GetCThreadTensorLengths_BM0_BM1_BN0_BN1();

        constexpr auto c_thread_desc_bm0_bm1_bn0_bn1 = make_naive_tensor_descriptor_packed(
            sequence_to_tuple_of_number(c_thread_tensor_lengths_bm0_bm1_bn0_bn1));

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_block_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size = math::integer_least_multiple(
            b_block_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block_double = p_shared_block;
        FloatAB* p_b_block_double = p_shared_block + 2 * a_block_aligned_space_size;

        // register allocation for output
        auto c_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAcc>(
            c_thread_desc_bm0_bm1_bn0_bn1.GetElementSpaceSize());

        ThreadwiseTensorSliceSet_v1<FloatAcc,
                                    decltype(c_thread_desc_bm0_bm1_bn0_bn1),
                                    decltype(c_thread_tensor_lengths_bm0_bm1_bn0_bn1)>{}
            .Run(c_thread_desc_bm0_bm1_bn0_bn1,
                 make_tuple(I0, I0, I0, I0),
                 c_thread_buf,
                 FloatAcc{0});

        constexpr auto a_block_slice_copy_step = make_multi_index(GK0PerBlock, 0, 0, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(GK0PerBlock, 0, 0, 0, 0);

        auto a_block_even_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_double, a_block_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize());
        auto b_block_even_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block_double, b_block_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize());

        auto a_block_odd_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_double + a_block_aligned_space_size,
            a_block_desc_gk0_gm0_gm10_gm11_gk1.GetElementSpaceSize());
        auto b_block_odd_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block_double + b_block_aligned_space_size,
            b_block_desc_gk0_gn0_gn10_gn11_gk1.GetElementSpaceSize());

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(
                a_grid_desc_gk0_gm0_gm10_gm11_gk1, a_global_buf, AGridStepHacks{});
            b_blockwise_copy.RunRead(
                b_grid_desc_gk0_gn0_gn10_gn11_gk1, b_global_buf, BGridStepHacks{});

            a_blockwise_copy.RunWrite(a_block_desc_gk0_gm0_gm10_gm11_gk1, a_block_even_buf);
            b_blockwise_copy.RunWrite(b_block_desc_gk0_gn0_gn10_gn11_gk1, b_block_even_buf);
        }

        if constexpr(HasMainKBlockLoop)
        {
            index_t gk0_block_on_grid = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowStepHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowStepHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_grid_desc_gk0_gm0_gm10_gm11_gk1, a_global_buf, AGridStepHacks{});
                b_blockwise_copy.RunRead(
                    b_grid_desc_gk0_gn0_gn10_gn11_gk1, b_global_buf, BGridStepHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(c_thread_desc_bm0_bm1_bn0_bn1,
                                   a_block_even_buf,
                                   b_block_even_buf,
                                   c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_block_desc_gk0_gm0_gm10_gm11_gk1, a_block_odd_buf);
                b_blockwise_copy.RunWrite(b_block_desc_gk0_gn0_gn10_gn11_gk1, b_block_odd_buf);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowStepHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowStepHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_grid_desc_gk0_gm0_gm10_gm11_gk1, a_global_buf, AGridStepHacks{});
                b_blockwise_copy.RunRead(
                    b_grid_desc_gk0_gn0_gn10_gn11_gk1, b_global_buf, BGridStepHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    c_thread_desc_bm0_bm1_bn0_bn1, a_block_odd_buf, b_block_odd_buf, c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_block_desc_gk0_gm0_gm10_gm11_gk1, a_block_even_buf);
                b_blockwise_copy.RunWrite(b_block_desc_gk0_gn0_gn10_gn11_gk1, b_block_even_buf);

                gk0_block_on_grid += 2 * GK0PerBlock;
            } while(gk0_block_on_grid < GK0 - 2 * GK0PerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_gk0_gm0_gm10_gm11_gk1,
                                                a_block_slice_copy_step,
                                                AGridMoveSliceWindowStepHacks{});
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_gk0_gn0_gn10_gn11_gk1,
                                                b_block_slice_copy_step,
                                                BGridMoveSliceWindowStepHacks{});

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(
                a_grid_desc_gk0_gm0_gm10_gm11_gk1, a_global_buf, AGridStepHacks{});
            b_blockwise_copy.RunRead(
                b_grid_desc_gk0_gn0_gn10_gn11_gk1, b_global_buf, BGridStepHacks{});

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(
                c_thread_desc_bm0_bm1_bn0_bn1, a_block_even_buf, b_block_even_buf, c_thread_buf);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_block_desc_gk0_gm0_gm10_gm11_gk1, a_block_odd_buf);
            b_blockwise_copy.RunWrite(b_block_desc_gk0_gn0_gn10_gn11_gk1, b_block_odd_buf);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_thread_desc_bm0_bm1_bn0_bn1, a_block_odd_buf, b_block_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_thread_desc_bm0_bm1_bn0_bn1, a_block_even_buf, b_block_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr auto c_thread_desc_gm10_bm0_bm1_gn10_bn0_bn1 =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1,
                               Number<c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I0]>{},
                               Number<c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I1]>{},
                               I1,
                               Number<c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I2]>{},
                               Number<c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I3]>{}));

            const auto c_thread_origin_on_block_bm0_bm1_bn0_bn1 =
                blockwise_gemm.CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(
                    get_thread_local_1d_id());

            ThreadwiseTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_thread_desc_gm10_bm0_bm1_gn10_bn0_bn1),
                decltype(c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1),
                Sequence<1,
                         c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I0],
                         c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I1],
                         1,
                         c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I2],
                         c_thread_tensor_lengths_bm0_bm1_bn0_bn1[I3]>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                false>{c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
                       make_multi_index(igm10,
                                        c_thread_origin_on_block_bm0_bm1_bn0_bn1[I0],
                                        c_thread_origin_on_block_bm0_bm1_bn0_bn1[I1],
                                        ign10,
                                        c_thread_origin_on_block_bm0_bm1_bn0_bn1[I2],
                                        c_thread_origin_on_block_bm0_bm1_bn0_bn1[I3])}
                .Run(c_thread_desc_gm10_bm0_bm1_gn10_bn0_bn1,
                     make_tuple(I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_grid_desc_gm10_bm0_bm1_gn10_bn0_bn1,
                     c_grid_buf,
                     CGridStepHacks{});
        }
    }
};

} // namespace ck
#endif
