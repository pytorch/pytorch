// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename ReducePtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ReduceInElementwiseOperations,
          typename ReduceAccElementwiseOperations,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ReduceGridDescriptor_MBlock_MPerBlock,
          typename ComputeBasePrtOfBatch,
          typename Block2CTileMap,
          bool HasMainK0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_reduce_xdl_cshuffle_v1(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            ReducePtrsGlobal p_reduces_grid,
            const index_t batch_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const ReduceInElementwiseOperations reduce_in_element_ops,
            const ReduceAccElementwiseOperations reduce_out_element_ops,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            const ReduceGridDescriptor_MBlock_MPerBlock reduce_grid_desc_mblock_mperblock,
            const ComputeBasePrtOfBatch compute_base_ptr_of_batch_,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetBBasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetCBasePtr(g_idx)));

    static_for<0, p_reduces_grid.Size(), 1>{}([&](auto In) {
        const long_index_t d_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(compute_base_ptr_of_batch_.GetDBasePtr(g_idx, In)));
        p_reduces_grid(In) = p_reduces_grid(In) + d_batch_offset;
    });

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainK0BlockLoop>(p_a_grid + a_batch_offset,
                                                   p_b_grid + b_batch_offset,
                                                   p_c_grid + c_batch_offset,
                                                   p_reduces_grid,
                                                   p_shared,
                                                   a_element_op,
                                                   b_element_op,
                                                   c_element_op,
                                                   reduce_in_element_ops,
                                                   reduce_out_element_ops,
                                                   a_grid_desc_ak0_m_ak1,
                                                   b_grid_desc_bk0_n_bk1,
                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                   reduce_grid_desc_mblock_mperblock,
                                                   block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = p_reduces_grid;
    ignore = batch_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = reduce_in_element_ops;
    ignore = reduce_out_element_ops;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = reduce_grid_desc_mblock_mperblock;
    ignore = compute_base_ptr_of_batch_;
    ignore = block_2_ctile_map;
#endif
}

// Note: inter-wave loop scheduler is rolled out to c-shuffle version first. Becuase non c-shuffle
// version currently has compiler issues with register spill which further causes validation
// failures.
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType,
          typename ReducePtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ReduceOperations,
          typename ReduceInElementwiseOperations,
          typename ReduceAccElementwiseOperations,
          typename ReduceGlobalMemoryDataOperation,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceBatchedGemmReduce_Xdl_CShuffle : public DeviceGemmReduce<0, ReduceOperations::Size()>
{
    using DeviceOp = DeviceBatchedGemmReduce_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(MRaw, MPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(M)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_right_pad_transform(MRaw, MPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(NRaw, NPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(N)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_right_pad_transform(NRaw, NPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    // assume D is packed tensor
    static auto MakeReduceGridDescriptor_M(index_t MRaw)
    {
        const auto d_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(d_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return d_grid_desc_mraw;
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using ReduceGridDesc_M    = decltype(MakeReduceGridDescriptor_M(1));

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA,
                                     index_t BatchStrideB,
                                     index_t BatchStrideC,
                                     index_t BatchStrideD)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideC_(BatchStrideC),
              BatchStrideD_(BatchStrideD)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        template <index_t I>
        __host__ __device__ constexpr long_index_t GetDBasePtr(index_t g_idx,
                                                               Number<I> reduction_idx) const
        {
            // TODO - Support sequence of StrideD in MakeArgument()
            (void)reduction_idx;
            return g_idx * static_cast<long_index_t>(BatchStrideD_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
        index_t BatchStrideD_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmReduce_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        ReduceAccDataType,
        ReducePtrsGlobal,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        ReduceOperations,
        ReduceInElementwiseOperations,
        ReduceAccElementwiseOperations,
        InMemoryDataOperationEnum::Set,
        ReduceGlobalMemoryDataOperation,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        ReduceGridDesc_M,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        CReduceThreadClusterLengths_MPerBlock_NPerBlock,
        CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
        CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock,
        LoopSched>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 ReducePtrsGlobal p_reduces_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 ReduceInElementwiseOperations reduce_in_element_ops,
                 ReduceAccElementwiseOperations reduce_out_element_ops,
                 index_t Batch)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              p_reduces_grid_{p_reduces_grid},
              Batch_(Batch),
              a_grid_desc_ak0_m_ak1_{DeviceOp::MakeAGridDescriptor_AK0_M_AK1(MRaw, KRaw, StrideA)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1(KRaw, NRaw, StrideB)},
              c_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC)},
              reduce_grid_desc_m_{DeviceOp::MakeReduceGridDescriptor_M(MRaw)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              reduce_grid_desc_mblock_mperblock_{},
              compute_base_ptr_of_batch_{
                  type_convert<index_t>(a_grid_desc_ak0_m_ak1_.GetElementSpaceSize()),
                  type_convert<index_t>(b_grid_desc_bk0_n_bk1_.GetElementSpaceSize()),
                  type_convert<index_t>(c_grid_desc_m_n_.GetElementSpaceSize()),
                  type_convert<index_t>(reduce_grid_desc_m_.GetElementSpaceSize())},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              reduce_in_element_ops_{reduce_in_element_ops},
              reduce_out_element_ops_{reduce_out_element_ops}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);

                reduce_grid_desc_mblock_mperblock_ =
                    GridwiseGemm::MakeReduceGridDescriptor_MBlock_MPerBlock(reduce_grid_desc_m_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        ReducePtrsGlobal p_reduces_grid_;
        index_t Batch_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        ReduceGridDesc_M reduce_grid_desc_m_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock
            reduce_grid_desc_mblock_mperblock_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        ReduceInElementwiseOperations reduce_in_element_ops_;
        ReduceAccElementwiseOperations reduce_out_element_ops_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if DEBUG_LOG
            {
                std::cout << "arg.Batch_ = " << arg.Batch_ << std::endl;

                std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                std::cout << "arg.reduce_grid_desc_m_{ " << arg.reduce_grid_desc_m_.GetLength(I0)
                          << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                            arg.b_grid_desc_bk0_n_bk1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Batch_;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            float elapsed_time = 0.0f;
            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_batched_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    ReducePtrsGlobal,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    ReduceInElementwiseOperations,
                    ReduceAccElementwiseOperations,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock,
                    ComputeBasePtrOfStridedBatch,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    true>;

                elapsed_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_reduces_grid_,
                                           arg.Batch_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.reduce_in_element_ops_,
                                           arg.reduce_out_element_ops_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.reduce_grid_desc_mblock_mperblock_,
                                           arg.compute_base_ptr_of_batch_,
                                           arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    ReducePtrsGlobal,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    ReduceInElementwiseOperations,
                    ReduceAccElementwiseOperations,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock,
                    ComputeBasePtrOfStridedBatch,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    false>;

                elapsed_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_reduces_grid_,
                                           arg.Batch_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.reduce_in_element_ops_,
                                           arg.reduce_out_element_ops_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.reduce_grid_desc_mblock_mperblock_,
                                           arg.compute_base_ptr_of_batch_,
                                           arg.block_2_ctile_map_);
            }

            return elapsed_time;
        }

        // polymorphic
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        auto casted_p_arg = dynamic_cast<const Argument*>(p_arg);
        if(casted_p_arg == nullptr)
        {
            return false;
        }
        else
        {
            return IsSupportedArgument(*casted_p_arg);
        }
    }

    static constexpr int NumReduce = ReduceOperations::Size();
    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             const void* p_bias,
                             std::array<const void*, 0> p_ds,
                             void* p_c,
                             std::array<void*, NumReduce> p_reduces,
                             ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideC,
                             std::array<ck::index_t, 0> StrideDs,
                             std::array<void*, 3> gemm_element_ops,
                             std::array<void*, 0> d_element_ops,
                             std::array<void*, NumReduce> reduce_in_element_op,
                             std::array<void*, NumReduce> reduce_out_element_op,
                             index_t Batch)
    {
        (void)p_bias;
        (void)p_ds;
        (void)StrideDs;
        (void)d_element_ops;

        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));

        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        static_cast<CDataType*>(p_c),
                        reduce_tuple,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        reduce_in_element_ops,
                        reduce_out_element_ops,
                        Batch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        std::array<const void*, 0> p_ds,
                        void* p_c,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, 0> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, 0> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_op,
                        std::array<void*, NumReduce> reduce_out_element_op,
                        index_t Batch = 1) override
    {
        (void)p_bias;
        (void)p_ds;
        (void)StrideDs;
        (void)d_element_ops;

        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));

        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          reduce_tuple,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          reduce_in_element_ops,
                                          reduce_out_element_ops,
                                          Batch);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemmReduce_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
