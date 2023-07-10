// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename A0B0B1DataType,
          typename D0sPointer,
          typename D1sPointer,
          typename E1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename CDE0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          typename A0GridDesc_AK0_M_AK1,
          typename B0GridDesc_BK0_N_BK1,
          typename D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5,
          typename B1GridDesc_BK0_N_BK1,
          typename D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2E1TileMap,
          typename ComputeBasePtrOfStridedBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_gemm_xdl_cshuffle_v1(
            const A0B0B1DataType* __restrict__ p_a0_grid,
            const A0B0B1DataType* __restrict__ p_b0_grid,
            D0sPointer p_d0s_grid,
            const A0B0B1DataType* __restrict__ p_b1_grid,
            D1sPointer p_d1s_grid,
            E1DataType* __restrict__ p_e1_grid,
            const A0ElementwiseOperation a0_element_op,
            const B0ElementwiseOperation b0_element_op,
            const CDE0ElementwiseOperation cde0_element_op,
            const B1ElementwiseOperation b1_element_op,
            const CDE1ElementwiseOperation cde1_element_op,
            const A0GridDesc_AK0_M_AK1 a0_grid_desc_ak0_m_ak1,
            const B0GridDesc_BK0_N_BK1 b0_grid_desc_bk0_n_bk1,
            const D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
                d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
            const B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1,
            const D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                d1s_grid_desc_mblock_mperblock_nblock_nperblock,
            const E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e1_grid_desc_mblock_mperblock_nblock_nperblock,
            const Block2E1TileMap block_2_e1tile_map,
            const index_t batch_count,
            const ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetBBasePtr(g_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetCBasePtr(g_idx)));

    static_for<0, p_d0s_grid.Size(), 1>{}([&](auto In) {
        const long_index_t d0_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(compute_base_ptr_of_batch.GetD0BasePtr(g_idx, In)));
        p_d0s_grid(In) = p_d0s_grid(In) + d0_batch_offset;
    });

    static_for<0, p_d1s_grid.Size(), 1>{}([&](auto In) {
        const long_index_t d1_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(compute_base_ptr_of_batch.GetD1BasePtr(g_idx, In)));
        p_d1s_grid(In) = p_d1s_grid(In) + d1_batch_offset;
    });

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a0_grid + a_batch_offset,
                                                  p_b0_grid + b_batch_offset,
                                                  p_d0s_grid,
                                                  p_b1_grid + b1_batch_offset,
                                                  p_d1s_grid,
                                                  p_e1_grid + c_batch_offset,
                                                  p_shared,
                                                  a0_element_op,
                                                  b0_element_op,
                                                  cde0_element_op,
                                                  b1_element_op,
                                                  cde1_element_op,
                                                  a0_grid_desc_ak0_m_ak1,
                                                  b0_grid_desc_bk0_n_bk1,
                                                  d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                                                  b1_grid_desc_bk0_n_bk1,
                                                  d1s_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e1_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_e1tile_map);
#else
    ignore = p_a0_grid;
    ignore = p_b0_grid;
    ignore = p_d0s_grid;
    ignore = p_b1_grid;
    ignore = p_d1s_grid;
    ignore = p_e1_grid;
    ignore = a0_element_op;
    ignore = b0_element_op;
    ignore = cde0_element_op;
    ignore = b1_element_op;
    ignore = cde1_element_op;
    ignore = a0_grid_desc_ak0_m_ak1;
    ignore = b0_grid_desc_bk0_n_bk1;
    ignore = d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5;
    ignore = b1_grid_desc_bk0_n_bk1;
    ignore = d1s_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e1_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_e1tile_map;
    ignore = batch_count;
    ignore = compute_base_ptr_of_batch;
#endif
}

// Computes C = A * B0 * B1
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <typename A0Layout,
          typename B0Layout, // B0Layout
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename A0DataType,
          typename B0DataType,
          typename Acc0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename Acc1DataType,
          typename C1ShuffleDataType,
          typename D1sDataType,
          typename E1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename CDE0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          bool PadGemm0M,
          bool PadGemm0N,
          bool PadGemm0K,
          bool PadGemm1N,
          bool PadGemm1K,
          index_t NumGemm0KPrefetchStage,
          index_t BlockSize,
          index_t Gemm0MPerBlock,
          index_t Gemm0NPerBlock,
          index_t Gemm0KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t A0K1,
          index_t B0K1,
          index_t B1K1,
          index_t Gemm0MPerXdl,
          index_t Gemm0NPerXdl,
          index_t Gemm0MXdlPerWave,
          index_t Gemm0NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          typename A0BlockTransferThreadClusterLengths_AK0_M_AK1,
          typename A0BlockTransferThreadClusterArrangeOrder,
          typename A0BlockTransferSrcAccessOrder,
          index_t A0BlockTransferSrcVectorDim,
          index_t A0BlockTransferSrcScalarPerVector,
          index_t A0BlockTransferDstScalarPerVector_AK1,
          bool A0BlockLdsExtraM,
          typename B0BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          index_t B0BlockTransferSrcVectorDim,
          index_t B0BlockTransferSrcScalarPerVector,
          index_t B0BlockTransferDstScalarPerVector_BK1,
          bool B0BlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1BlockLdsExtraN,
          index_t C1ShuffleMXdlPerWavePerShuffle,
          index_t C1ShuffleGemm0NXdlPerWavePerShuffle,
          typename CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = LoopScheduler::Default>
struct DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle
    : public DeviceBatchedGemmMultipleDGemmMultipleD<A0Layout,
                                                     B0Layout,
                                                     D0sLayout,
                                                     B1Layout,
                                                     D1sLayout,
                                                     E1Layout,
                                                     A0DataType,
                                                     B0DataType,
                                                     D0sDataType,
                                                     B1DataType,
                                                     D1sDataType,
                                                     E1DataType,
                                                     A0ElementwiseOperation,
                                                     B0ElementwiseOperation,
                                                     CDE0ElementwiseOperation,
                                                     B1ElementwiseOperation,
                                                     CDE1ElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle;

    static constexpr index_t NumD0Tensor = D0sDataType::Size();
    static constexpr index_t NumD1Tensor = D1sDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};
    static constexpr auto I9 = Number<9>{};

    static constexpr auto gemm0_padder =
        GemmPadder_v2<PadGemm0M, PadGemm0N, PadGemm0K, index_t, index_t, index_t>{
            Gemm0MPerBlock, Gemm0NPerBlock, Gemm0KPerBlock};

    static constexpr auto gemm1_padder =
        GemmPadder_v2<PadGemm0M, PadGemm1N, PadGemm1K, index_t, index_t, index_t>{
            Gemm0MPerBlock, Gemm1NPerBlock, Gemm1KPerBlock};

    // for Gemm0
    static auto MakeA0GridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA0)
    {
        const auto a0_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, A0Layout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA0, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, A0Layout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA0));
            }
        }();

        return gemm0_padder.PadADescriptor_M_K(a0_grid_desc_mraw_kraw);
    }

    // for Gemm0
    static auto MakeB0GridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b0_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, B0Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, B0Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return gemm0_padder.PadBDescriptor_N_K(b0_grid_desc_nraw_kraw);
    }

    // for Gemm0
    template <typename DLay>
    static auto MakeD0GridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideD0)
    {
        const auto d0_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, DLay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideD0, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, DLay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideD0));
            }
        }();

        return gemm0_padder.PadCDescriptor_M_N(d0_grid_desc_mraw_nraw);
    }

    // for Gemm1
    static auto MakeB1GridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b1_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, B1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, B1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return gemm1_padder.PadBDescriptor_N_K(b1_grid_desc_nraw_kraw);
    }

    // for Gemm1
    template <typename ELay>
    static auto MakeE1GridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE1)
    {
        const auto e1_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE1, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE1));
            }
        }();

        return gemm1_padder.PadCDescriptor_M_N(e1_grid_desc_mraw_nraw);
    }

    static auto MakeD0sGridDescriptor_M_N(const std::array<index_t, NumD1Tensor>& MRaws,
                                          const std::array<index_t, NumD1Tensor>& NRaws,
                                          const std::array<index_t, NumD1Tensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, D0sLayout>>;

                return DeviceOp::MakeD0GridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumD0Tensor>{});
    }

    static auto MakeD1sGridDescriptor_M_N(const std::array<index_t, NumD1Tensor>& MRaws,
                                          const std::array<index_t, NumD1Tensor>& NRaws,
                                          const std::array<index_t, NumD1Tensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, D1sLayout>>;

                return DeviceOp::MakeE1GridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumD1Tensor>{});
    }

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA0,
                                     index_t BatchStrideB0,
                                     std::array<index_t, NumD0Tensor> BatchStrideD0s,
                                     index_t BatchStrideB1,
                                     std::array<index_t, NumD1Tensor> BatchStrideD1s,
                                     index_t BatchStrideE1)
            : BatchStrideA0_(BatchStrideA0),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideD0s_(BatchStrideD0s),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideD1s_(BatchStrideD1s),
              BatchStrideE1_(BatchStrideE1)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA0_);
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB0_);
        }

        template <index_t I>
        __host__ __device__ constexpr long_index_t GetD0BasePtr(index_t g_idx,
                                                                Number<I> d1_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD0s_[d1_idx]);
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB1_);
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideE1_);
        }

        template <index_t I>
        __host__ __device__ constexpr auto GetD1BasePtr(index_t g_idx, Number<I> d1_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD1s_[d1_idx]);
        }

        private:
        index_t BatchStrideA0_;
        index_t BatchStrideB0_;
        std::array<index_t, NumD0Tensor> BatchStrideD0s_;
        index_t BatchStrideB1_;
        std::array<index_t, NumD1Tensor> BatchStrideD1s_;
        index_t BatchStrideE1_;
    };

    using A0GridDesc_M_K  = decltype(MakeA0GridDescriptor_M_K(1, 1, 1));
    using B0GridDesc_N_K  = decltype(MakeB0GridDescriptor_N_K(1, 1, 1));
    using D0sGridDesc_M_N = remove_cvref_t<decltype(MakeD0sGridDescriptor_M_N({}, {}, {}))>;
    using B1GridDesc_N_K  = decltype(MakeB1GridDescriptor_N_K(1, 1, 1));
    using D1sGridDesc_M_N = remove_cvref_t<decltype(MakeD1sGridDescriptor_M_N({}, {}, {}))>;
    using E1GridDesc_M_N  = decltype(MakeE1GridDescriptor_M_N<E1Layout>(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle<
        A0DataType, // TODO: distinguish A/B datatype
        Acc0DataType,
        D0sDataType,
        Acc1DataType,
        C1ShuffleDataType,
        D1sDataType,
        E1DataType,
        A0ElementwiseOperation,
        B0ElementwiseOperation,
        CDE0ElementwiseOperation,
        B1ElementwiseOperation,
        CDE1ElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        A0GridDesc_M_K,
        B0GridDesc_N_K,
        D0sGridDesc_M_N,
        B1GridDesc_N_K,
        D1sGridDesc_M_N,
        E1GridDesc_M_N,
        NumGemm0KPrefetchStage,
        BlockSize,
        Gemm0MPerBlock,
        Gemm0NPerBlock,
        Gemm0KPerBlock,
        Gemm1NPerBlock,
        Gemm1KPerBlock,
        A0K1,
        B0K1,
        B1K1,
        Gemm0MPerXdl,
        Gemm0NPerXdl,
        Gemm0MXdlPerWave,
        Gemm0NXdlPerWave,
        Gemm1NXdlPerWave,
        A0BlockTransferThreadClusterLengths_AK0_M_AK1,
        A0BlockTransferThreadClusterArrangeOrder,
        A0BlockTransferSrcAccessOrder,
        A0BlockTransferSrcVectorDim,
        A0BlockTransferSrcScalarPerVector,
        A0BlockTransferDstScalarPerVector_AK1,
        true,
        A0BlockLdsExtraM,
        B0BlockTransferThreadClusterLengths_BK0_N_BK1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_BK1,
        true,
        B0BlockLdsExtraN,
        B1BlockTransferThreadClusterLengths_BK0_N_BK1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_BK1,
        false,
        B1BlockLdsExtraN,
        C1ShuffleMXdlPerWavePerShuffle,
        C1ShuffleGemm0NXdlPerWavePerShuffle,
        CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    using A0GridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultA0GridDescriptor_AK0_M_AK1(A0GridDesc_M_K{}))>;
    using B0GridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultB0GridDescriptor_BK0_N_BK1(B0GridDesc_N_K{}))>;
    using B1GridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultB1GridDescriptor_BK0_N_BK1(B1GridDesc_N_K{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const A0DataType* p_a0_grid,
                 const B0DataType* p_b0_grid,
                 std::array<const void*, NumD0Tensor> p_d0s_grid,
                 const B1DataType* p_b1_grid,
                 std::array<const void*, NumD1Tensor> p_d1s_grid,
                 E1DataType* p_e1_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t Gemm1NRaw, // = ORaw
                 index_t Batch,
                 index_t StrideA0,
                 index_t StrideB0,
                 std::array<index_t, NumD0Tensor> StrideD0s,
                 index_t StrideB1,
                 std::array<index_t, NumD1Tensor> StrideD1s,
                 index_t StrideE1,
                 index_t BatchStrideA0,
                 index_t BatchStrideB0,
                 std::array<index_t, NumD0Tensor> BatchStrideD0s,
                 index_t BatchStrideB1,
                 std::array<index_t, NumD1Tensor> BatchStrideD1s,
                 index_t BatchStrideE1,
                 A0ElementwiseOperation a0_element_op,
                 B0ElementwiseOperation b0_element_op,
                 CDE0ElementwiseOperation cde0_element_op,
                 B1ElementwiseOperation b1_element_op,
                 CDE1ElementwiseOperation cde1_element_op)
            : p_a0_grid_{p_a0_grid},
              p_b0_grid_{p_b0_grid},
              p_d0s_grid_{},
              p_b1_grid_{p_b1_grid},
              p_d1s_grid_{},
              p_e1_grid_{p_e1_grid},
              a0_grid_desc_m_k_{DeviceOp::MakeA0GridDescriptor_M_K(MRaw, KRaw, StrideA0)},
              b0_grid_desc_n_k_{DeviceOp::MakeB0GridDescriptor_N_K(KRaw, NRaw, StrideB0)},
              d0s_grid_desc_m_n_{},
              b1_grid_desc_n_k_{DeviceOp::MakeB1GridDescriptor_N_K(NRaw, Gemm1NRaw, StrideB1)},
              d1s_grid_desc_m_n_{},
              e1_grid_desc_m_n_{
                  DeviceOp::MakeE1GridDescriptor_M_N<E1Layout>(MRaw, Gemm1NRaw, StrideE1)},
              a0_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultA0GridDescriptor_AK0_M_AK1(a0_grid_desc_m_k_)},
              b0_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultB0GridDescriptor_BK0_N_BK1(b0_grid_desc_n_k_)},
              d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_{},
              b1_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultB1GridDescriptor_BK0_N_BK1(b1_grid_desc_n_k_)},
              d1s_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e1_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_e1tile_map_{GridwiseGemm::MakeDefaultBlock2E1TileMap(e1_grid_desc_m_n_)},
              a0_element_op_{a0_element_op},
              b0_element_op_{b0_element_op},
              cde0_element_op_{cde0_element_op},
              b1_element_op_{b1_element_op},
              cde1_element_op_{cde1_element_op},
              batch_count_(Batch),
              compute_base_ptr_of_batch_{BatchStrideA0,
                                         BatchStrideB0,
                                         BatchStrideD0s,
                                         BatchStrideB1,
                                         BatchStrideD1s,
                                         BatchStrideE1}
        {
#if DEBUG_LOG
            std::cout << "a0_grid_desc_m_k_{" << a0_grid_desc_m_k_.GetLength(I0) << ", "
                      << a0_grid_desc_m_k_.GetLength(I1) << "}" << std::endl;
            std::cout << "b0_grid_desc_n_k_{" << b0_grid_desc_n_k_.GetLength(I0) << ", "
                      << b0_grid_desc_n_k_.GetLength(I1) << "}" << std::endl;
            std::cout << "d0s_grid_desc_m_n_[I0]{" << d0s_grid_desc_m_n_[I0].GetLength(I0) << ", "
                      << d0s_grid_desc_m_n_[I0].GetLength(I1) << "}" << std::endl;
            std::cout << "b1_grid_desc_n_k_{" << b1_grid_desc_n_k_.GetLength(I0) << ", "
                      << b1_grid_desc_n_k_.GetLength(I1) << "}" << std::endl;
            std::cout << "d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_{"
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I0) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I1) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I2) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I3) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I4) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I5) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I6) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I7) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I8) << ", "
                      << d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_[I0].GetLength(I9) << "}"
                      << std::endl;
            std::cout << "e1_grid_desc_m_n_{" << e1_grid_desc_m_n_.GetLength(I0) << ", "
                      << e1_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
#endif

            static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                using D0Layout   = remove_cvref_t<tuple_element_t<i.value, D0sLayout>>;
                using D0DataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;

                // D0 pointer
                p_d0s_grid_(i) = static_cast<const D0DataType*>(p_d0s_grid[i]);

                // D0 desc
                d0s_grid_desc_m_n_(i) =
                    DeviceOp::MakeD0GridDescriptor_M_N<D0Layout>(MRaw, NRaw, StrideD0s[i]);
            });

            static_for<0, NumD1Tensor, 1>{}([&](auto i) {
                using D1Layout   = remove_cvref_t<tuple_element_t<i.value, D1sLayout>>;
                using D1DataType = remove_cvref_t<tuple_element_t<i.value, D1sDataType>>;

                // D1 pointer
                p_d1s_grid_(i) = static_cast<const D1DataType*>(p_d1s_grid[i]);

                // D1 desc
                d1s_grid_desc_m_n_(i) =
                    DeviceOp::MakeE1GridDescriptor_M_N<D1Layout>(MRaw, Gemm1NRaw, StrideD1s[i]);
            });

            if(GridwiseGemm::CheckValidity(a0_grid_desc_m_k_,
                                           b0_grid_desc_n_k_,
                                           b1_grid_desc_n_k_,
                                           e1_grid_desc_m_n_,
                                           block_2_e1tile_map_))
            {
                e1_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e1_grid_desc_m_n_);

                d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_ =
                    GridwiseGemm::MakeD0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
                        d0s_grid_desc_m_n_);

                d1s_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        d1s_grid_desc_m_n_);
            }
        }

        //  private:
        // pointers
        const A0DataType* p_a0_grid_;
        const B0DataType* p_b0_grid_;
        typename GridwiseGemm::D0sGridPointer p_d0s_grid_;
        const B1DataType* p_b1_grid_;
        typename GridwiseGemm::D1sGridPointer p_d1s_grid_;
        E1DataType* p_e1_grid_;

        // tensor descriptors for problem definiton
        A0GridDesc_M_K a0_grid_desc_m_k_;
        B0GridDesc_N_K b0_grid_desc_n_k_;
        D0sGridDesc_M_N d0s_grid_desc_m_n_;
        B1GridDesc_N_K b1_grid_desc_n_k_;
        D1sGridDesc_M_N d1s_grid_desc_m_n_;
        E1GridDesc_M_N e1_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        A0GridDesc_AK0_M_AK1 a0_grid_desc_ak0_m_ak1_;
        B0GridDesc_BK0_N_BK1 b0_grid_desc_bk0_n_bk1_;
        typename GridwiseGemm::D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
            d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_;
        B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1_;
        typename GridwiseGemm::D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            d1s_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e1_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e1-tile map
        typename GridwiseGemm::DefaultBlock2E1TileMap block_2_e1tile_map_;

        // element-wise op
        A0ElementwiseOperation a0_element_op_;
        B0ElementwiseOperation b0_element_op_;
        CDE0ElementwiseOperation cde0_element_op_;
        B1ElementwiseOperation b1_element_op_;
        CDE1ElementwiseOperation cde1_element_op_;

        // batch
        index_t batch_count_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a0_grid_desc_m_k_,
                                            arg.b0_grid_desc_n_k_,
                                            arg.b1_grid_desc_n_k_,
                                            arg.e1_grid_desc_m_n_,
                                            arg.block_2_e1tile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_e1tile_map_.CalculateGridSize(arg.e1_grid_desc_m_n_) * arg.batch_count_;

            // Gemm0_K
            const auto K = arg.a0_grid_desc_m_k_.GetLength(I1);

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_batched_gemm_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    A0DataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::D0sGridPointer,
                    typename GridwiseGemm::D1sGridPointer,
                    E1DataType,
                    A0ElementwiseOperation,
                    B0ElementwiseOperation,
                    CDE0ElementwiseOperation,
                    B1ElementwiseOperation,
                    CDE1ElementwiseOperation,
                    DeviceOp::A0GridDesc_AK0_M_AK1,
                    DeviceOp::B0GridDesc_BK0_N_BK1,
                    typename GridwiseGemm::D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5,
                    DeviceOp::B1GridDesc_BK0_N_BK1,
                    typename GridwiseGemm::D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DefaultBlock2E1TileMap,
                    ComputeBasePtrOfStridedBatch,
                    has_main_k_block_loop_>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a0_grid_,
                                              arg.p_b0_grid_,
                                              arg.p_d0s_grid_,
                                              arg.p_b1_grid_,
                                              arg.p_d1s_grid_,
                                              arg.p_e1_grid_,
                                              arg.a0_element_op_,
                                              arg.b0_element_op_,
                                              arg.cde0_element_op_,
                                              arg.b1_element_op_,
                                              arg.cde1_element_op_,
                                              arg.a0_grid_desc_ak0_m_ak1_,
                                              arg.b0_grid_desc_bk0_n_bk1_,
                                              arg.d0s_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
                                              arg.b1_grid_desc_bk0_n_bk1_,
                                              arg.d1s_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e1_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_e1tile_map_,
                                              arg.batch_count_,
                                              arg.compute_base_ptr_of_batch_);
            };

            // Gemm1_K is split into Gemm1_K0/K1 where K1 is known at compile time, so we only need
            // to concern Gemm0's loop
            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
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
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg.a0_grid_desc_m_k_,
                                           arg.b0_grid_desc_n_k_,
                                           arg.b1_grid_desc_n_k_,
                                           arg.e1_grid_desc_m_n_,
                                           arg.block_2_e1tile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const A0DataType* p_a0,
                             const B0DataType* p_b0,
                             std::array<const void*, NumD0Tensor> p_d0s,
                             const B1DataType* p_b1,
                             std::array<const void*, NumD1Tensor> p_d1s,
                             E1DataType* p_e1,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t Gemm1NRaw,
                             index_t Batch,
                             index_t StrideA0,
                             index_t StrideB0,
                             std::array<index_t, NumD0Tensor> StrideD0s,
                             index_t StrideB1,
                             std::array<index_t, NumD1Tensor> StrideD1s,
                             index_t StrideE1,
                             index_t BatchStrideA0,
                             index_t BatchStrideB0,
                             std::array<index_t, NumD0Tensor> BatchStrideD0s,
                             index_t BatchStrideB1,
                             std::array<index_t, NumD1Tensor> BatchStrideD1s,
                             index_t BatchStrideE1,
                             A0ElementwiseOperation a0_element_op,
                             B0ElementwiseOperation b0_element_op,
                             CDE0ElementwiseOperation cde0_element_op,
                             B1ElementwiseOperation b1_element_op,
                             CDE1ElementwiseOperation cde1_element_op)
    {
        return Argument{p_a0,          p_b0,
                        p_d0s,         p_b1,
                        p_d1s,         p_e1,
                        MRaw,          NRaw,
                        KRaw,          Gemm1NRaw,
                        Batch,         StrideA0,
                        StrideB0,      StrideD0s,
                        StrideB1,      StrideD1s,
                        StrideE1,      BatchStrideA0,
                        BatchStrideB0, BatchStrideD0s,
                        BatchStrideB1, BatchStrideD1s,
                        BatchStrideE1, a0_element_op,
                        b0_element_op, cde0_element_op,
                        b1_element_op, cde1_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a0,
                        const void* p_b0,
                        std::array<const void*, NumD0Tensor> p_d0s,
                        const void* p_b1,
                        std::array<const void*, NumD1Tensor> p_d1s,
                        void* p_e1,
                        index_t MRaw,
                        index_t NRaw,
                        index_t KRaw,
                        index_t Gemm1NRaw,
                        index_t Batch,
                        index_t StrideA0,
                        index_t StrideB0,
                        std::array<ck::index_t, NumD0Tensor> StrideD0s,
                        index_t StrideB1,
                        std::array<ck::index_t, NumD1Tensor> StrideD1s,
                        index_t StrideE1,
                        index_t BatchStrideA0,
                        index_t BatchStrideB0,
                        std::array<ck::index_t, NumD0Tensor> BatchStrideD0s,
                        index_t BatchStrideB1,
                        std::array<ck::index_t, NumD1Tensor> BatchStrideD1s,
                        index_t BatchStrideE1,
                        A0ElementwiseOperation a0_element_op,
                        B0ElementwiseOperation b0_element_op,
                        CDE0ElementwiseOperation cde0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CDE1ElementwiseOperation cde1_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const A0DataType*>(p_a0),
                                          static_cast<const B0DataType*>(p_b0),
                                          p_d0s,
                                          static_cast<const B1DataType*>(p_b1),
                                          p_d1s,
                                          static_cast<E1DataType*>(p_e1),
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          Gemm1NRaw,
                                          Batch,
                                          StrideA0,
                                          StrideB0,
                                          StrideD0s,
                                          StrideB1,
                                          StrideD1s,
                                          StrideE1,
                                          BatchStrideA0,
                                          BatchStrideB0,
                                          BatchStrideD0s,
                                          BatchStrideB1,
                                          BatchStrideD1s,
                                          BatchStrideE1,
                                          a0_element_op,
                                          b0_element_op,
                                          cde0_element_op,
                                          b1_element_op,
                                          cde1_element_op);
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
        str << "DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << Gemm0MPerBlock << ", "
            << Gemm0NPerBlock << ", "
            << Gemm0KPerBlock << ", "
            << A0K1 << ", "
            << B0K1 << ", "
            << B1K1 << ", "
            << Gemm0MPerXdl << ", "
            << Gemm0NPerXdl << ", "
            << Gemm0MXdlPerWave << ", "
            << Gemm0NXdlPerWave << ", "
            << Gemm1NXdlPerWave << "> ";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
