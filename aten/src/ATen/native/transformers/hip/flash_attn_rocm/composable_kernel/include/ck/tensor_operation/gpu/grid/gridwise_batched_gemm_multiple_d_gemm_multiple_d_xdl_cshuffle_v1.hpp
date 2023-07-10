// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename A0B0B1DataType, // FIXME: don't assume A0/B0/B1 have same datatype
          typename Acc0DataType,
          typename D0sDataType,
          typename Acc1DataType,
          typename C1ShuffleDataType,
          typename D1sDataType,
          typename E1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename CDE0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          InMemoryDataOperationEnum E1GlobalMemoryDataOperation,
          typename A0GridDesc_M_K,
          typename B0GridDesc_N_K,
          typename D0sGridDesc_M_N,
          typename B1GridDesc_N_K,
          typename D1sGridDesc_M_N,
          typename E1GridDesc_M_N,
          index_t NumGemm0KPrefetchStage,
          index_t BlockSize,
          index_t Gemm0MPerBlock,
          index_t Gemm0NPerBlock,
          index_t Gemm0KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t A0K1Value,
          index_t B0K1Value,
          index_t B1K1Value,
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
          bool A0ThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t A0BlockLdsExtraM,
          typename B0BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          index_t B0BlockTransferSrcVectorDim,
          index_t B0BlockTransferSrcScalarPerVector,
          index_t B0BlockTransferDstScalarPerVector_BK1,
          bool B0ThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t B0BlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          index_t B1BlockLdsExtraN,
          index_t C1ShuffleGemm0MXdlPerWavePerShuffle,
          index_t C1ShuffleGemm0NXdlPerWavePerShuffle,
          typename CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched>
struct GridwiseBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle
{
    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

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

    static constexpr auto WaveSize = 64;
    // K1 should be Number<...>
    // Gemm0
    static constexpr auto A0K1 = Number<A0K1Value>{};
    static constexpr auto B0K1 = Number<B0K1Value>{};

    static constexpr auto A0K0PerBlock = Number<Gemm0KPerBlock / A0K1Value>{};
    static constexpr auto B0K0PerBlock = Number<Gemm0KPerBlock / B0K1Value>{};

    static constexpr auto Gemm0MWaves = Gemm0MPerBlock / (Gemm0MPerXdl * Gemm0MXdlPerWave);
    static constexpr auto Gemm0NWaves = Gemm0NPerBlock / (Gemm0NPerXdl * Gemm0NXdlPerWave);
    // Gemm1
    static constexpr auto B1K1         = Number<B1K1Value>{};
    static constexpr auto B1K0PerBlock = Number<Gemm1KPerBlock / B1K1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = GridwiseGemmPipeline_v1<NumGemm0KPrefetchStage>;

    // ck::Tuple<const D0DataType1*, const D0DataType2*, ...>
    static constexpr auto MakeD0sGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using D0DataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;

                return static_cast<const D0DataType*>(nullptr);
            },
            Number<NumD0Tensor>{});
    }

    // ck::Tuple<const D1DataType1*, const D1DataType2*, ...>
    static constexpr auto MakeD1sGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using D1DataType = remove_cvref_t<tuple_element_t<i.value, D1sDataType>>;

                return static_cast<const D1DataType*>(nullptr);
            },
            Number<NumD1Tensor>{});
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(Gemm0MWaves, Gemm0NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / Gemm0NPerXdl, Gemm0NPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    template <typename A0BlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const A0BlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = Gemm0MPerBlock / (Gemm0MXdlPerWave * Gemm0MPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm0MXdlPerWave, MWaves, Gemm0MPerXdl>(
            A0BlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = Gemm0NPerBlock / (Gemm0NXdlPerWave * Gemm0NPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm0NXdlPerWave, NWaves, Gemm0NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    template <typename A0BlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const A0BlockDesc_AK0_M_AK1&)
    {
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm0MXdlPerWave, 1, 1>(
            A0BlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * Gemm0NPerXdl);
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, Gemm0NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static constexpr auto GetA0BlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A0 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(A0K0PerBlock, Number<Gemm0MPerBlock>{}, A0K1),
            make_tuple(Number<Gemm0MPerBlock + A0BlockLdsExtraM>{} * A0K1, A0K1, I1));
    }

    __host__ __device__ static constexpr auto GetB0BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B0 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B0K0PerBlock, Number<Gemm0NPerBlock>{}, B0K1),
            make_tuple(Number<Gemm0NPerBlock + B0BlockLdsExtraN>{} * B0K1, B0K1, I1));
    }

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0PerBlock, Number<Gemm1NPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1NPerBlock + B1BlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetC1ShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = Gemm0MPerBlock / (Gemm0MXdlPerWave * Gemm0MPerXdl);
        constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * Gemm0NPerXdl);

        constexpr auto c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<C1ShuffleGemm0MXdlPerWavePerShuffle * MWave * Gemm0MPerXdl>{},
                           I1,
                           Number<C1ShuffleGemm0NXdlPerWavePerShuffle * NWave * Gemm0NPerXdl>{}));

        return c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a0_block_space_size_aligned +
                                         SharedMemTrait::b0_block_space_size_aligned) *
                                        sizeof(A0B0B1DataType);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(A0B0B1DataType);
        const index_t c1_block_bytes_end =
            SharedMemTrait::c1_block_space_size * sizeof(C1ShuffleDataType);

        return math::max(gemm0_bytes_end, gemm1_bytes_end, c1_block_bytes_end);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2E1TileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const A0GridDesc_M_K& a0_grid_desc_m_k,
                  const B0GridDesc_N_K& b0_grid_desc_n_k,
                  const B1GridDesc_N_K& b1_grid_desc_n_k,
                  const E1GridDesc_M_N& e1_grid_desc_m_n,
                  const Block2E1TileMap& block_2_e1tile_map)
    {
        static_assert((Gemm0MPerBlock % (Gemm0MPerXdl * Gemm0MXdlPerWave) == 0) &&
                          (Gemm0NPerBlock % (Gemm0NXdlPerWave * Gemm0NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M      = a0_grid_desc_m_k.GetLength(I0);
        const auto N      = b0_grid_desc_n_k.GetLength(I0);
        const auto K      = a0_grid_desc_m_k.GetLength(I1);
        const auto Gemm1N = b1_grid_desc_n_k.GetLength(I0);

        if(!(M == e1_grid_desc_m_n.GetLength(I0) && Gemm1N == e1_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        if(!(M % Gemm0MPerBlock == 0 && N % Gemm0NPerBlock == 0 && K % Gemm0KPerBlock == 0 &&
             Gemm1N % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / Gemm0KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(Gemm0NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = Gemm0NPerBlock / Gemm1KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
        {
            return false;
        }

        if(!block_2_e1tile_map.CheckValidity(e1_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / Gemm0KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    // A0 desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultA0GridDescriptor_AK0_M_AK1(const A0GridDesc_M_K& a0_grid_desc_m_k)
    {
        const auto M = a0_grid_desc_m_k.GetLength(I0);
        const auto K = a0_grid_desc_m_k.GetLength(I1);

        const auto A0K0 = K / A0K1;

        return transform_tensor_descriptor(
            a0_grid_desc_m_k,
            make_tuple(make_unmerge_transform(make_tuple(A0K0, A0K1)),
                       make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // B0 desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultB0GridDescriptor_BK0_N_BK1(const B0GridDesc_N_K& b0_grid_desc_n_k)
    {
        const auto N = b0_grid_desc_n_k.GetLength(I0);
        const auto K = b0_grid_desc_n_k.GetLength(I1);

        const auto B0K0 = K / B0K1;

        return transform_tensor_descriptor(
            b0_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(B0K0, B0K1)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // D0 desc for source in blockwise copy
    template <typename D0GridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeGemm0D0GridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(const D0GridDesc_M_N& d0_grid_desc_m_n)
    {
        const auto M = d0_grid_desc_m_n.GetLength(I0);
        const auto N = d0_grid_desc_m_n.GetLength(I1);

        constexpr auto mfma =
            MfmaSelector<A0B0B1DataType, Gemm0MPerXdl, Gemm0NPerXdl>::selected_mfma;
        constexpr auto N3 = mfma.num_groups_per_blk;
        constexpr auto N5 = mfma.group_size;
        return transform_tensor_descriptor(
            d0_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(
                           M / Gemm0MPerBlock, Gemm0MXdlPerWave, Gemm0MWaves, Gemm0MPerXdl)),
                       make_unmerge_transform(make_tuple(N / Gemm0NPerBlock,
                                                         Gemm0NXdlPerWave,
                                                         Gemm0NWaves,
                                                         N3,
                                                         WaveSize / Gemm0NPerXdl,
                                                         N5))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7, 8, 9>{}));
    }

    // B1 desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultB1GridDescriptor_BK0_N_BK1(const B1GridDesc_N_K& b1_grid_desc_n_k)
    {
        const auto N = b1_grid_desc_n_k.GetLength(I0);
        const auto K = b1_grid_desc_n_k.GetLength(I1);

        const auto B1K0 = K / B1K1;

        return transform_tensor_descriptor(
            b1_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(B1K0, B1K1)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // C1 desc for destination in blockwise copy
    __host__ __device__ static constexpr auto
    MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const E1GridDesc_M_N& e1_grid_desc_m_n)
    {
        const auto M = e1_grid_desc_m_n.GetLength(I0);
        const auto N = e1_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / Gemm0MPerBlock;
        const auto NBlock = N / Gemm1NPerBlock;

        const auto e1_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e1_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<Gemm0MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e1_grid_desc_mblock_mperblock_nblock_nperblock;
    }
    // D0s desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeD0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(const D0sGridDesc_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeGemm0D0GridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(ds_grid_desc_m_n[i]);
            },
            Number<NumD0Tensor>{});
    }
    // Ds desc for source in blockwise copy
    template <typename DsGridDescriptor_M_N>
    __host__ __device__ static constexpr auto
    MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDescriptor_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumD1Tensor>{});
    }

    // return block_id to C1 matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2E1TileMap(const E1GridDesc_M_N& e1_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<Gemm0MPerBlock, Gemm1NPerBlock, E1GridDesc_M_N>(
            e1_grid_desc_m_n);
    }

    using E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(E1GridDesc_M_N{}))>;

    using D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5 = remove_cvref_t<decltype(
        MakeD0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(D0sGridDesc_M_N{}))>;

    using D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(D1sGridDesc_M_N{}))>;

    using DefaultBlock2E1TileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2E1TileMap(E1GridDesc_M_N{}))>;

    struct SharedMemTrait
    {
        // LDS allocation for A0 and B0: be careful of alignment
        static constexpr auto a0_block_desc_ak0_m_ak1 =
            GetA0BlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b0_block_desc_bk0_n_bk1 =
            GetB0BlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        static constexpr auto max_lds_align = math::lcm(math::lcm(A0K1, B0K1), B1K1);

        static constexpr auto a0_block_space_size_aligned = math::integer_least_multiple(
            a0_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b0_block_space_size_aligned = math::integer_least_multiple(
            b0_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a0_block_space_offset = 0;
        static constexpr auto b0_block_space_offset = a0_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = 0;

        // LDS allocation for C1 shuffle in LDS
        static constexpr auto c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetC1ShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c1_block_space_size =
            c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    using D0sGridPointer = decltype(MakeD0sGridPointer());
    using D1sGridPointer = decltype(MakeD1sGridPointer());

    template <bool HasMainKBlockLoop,
              typename A0GridDesc_AK0_M_AK1,
              typename B0GridDesc_BK0_N_BK1,
              typename B1GridDesc_BK0_N_BK1,
              typename Block2E1TileMap>
    __device__ static void Run(const A0B0B1DataType* __restrict__ p_a0_grid,
                               const A0B0B1DataType* __restrict__ p_b0_grid,
                               D0sGridPointer p_d0s_grid,
                               const A0B0B1DataType* __restrict__ p_b1_grid,
                               D1sGridPointer p_d1s_grid,
                               E1DataType* __restrict__ p_e1_grid,
                               void* __restrict__ p_shared,
                               const A0ElementwiseOperation& a0_element_op,
                               const B0ElementwiseOperation& b0_element_op,
                               const CDE0ElementwiseOperation& cde0_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CDE1ElementwiseOperation& cde1_element_op,
                               const A0GridDesc_AK0_M_AK1& a0_grid_desc_ak0_m_ak1,
                               const B0GridDesc_BK0_N_BK1& b0_grid_desc_bk0_n_bk1,
                               const D0sGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5&
                                   d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                               const B1GridDesc_BK0_N_BK1& b1_grid_desc_bk0_n_bk1,
                               const D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   d1s_grid_desc_mblock_mperblock_nblock_nperblock,
                               const E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e1_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2E1TileMap& block_2_e1tile_map)
    {
        const auto a0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a0_grid, a0_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b0_grid, b0_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        auto e1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e1_grid, e1_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());
        const auto d0s_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_d0s_grid[i],
                    d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i].GetElementSpaceSize());
            },
            Number<NumD0Tensor>{});
        const auto d1s_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_d1s_grid[i],
                    d1s_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumD1Tensor>{});

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_e1tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_e1tile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e1_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e1_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * Gemm0MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1NPerBlock);

        // A0 matrix in LDS memory, dst of blockwise copy
        constexpr auto a0_block_desc_ak0_m_ak1 = GetA0BlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B0 matrix in LDS memory, dst of blockwise copy
        constexpr auto b0_block_desc_bk0_n_bk1 = GetB0BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        //
        // set up Gemm0
        //

        // A0 matrix blockwise copy
        auto a0_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                A0ElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<A0K0PerBlock, Gemm0MPerBlock, A0K1>,
                                                A0BlockTransferThreadClusterLengths_AK0_M_AK1,
                                                A0BlockTransferThreadClusterArrangeOrder,
                                                A0B0B1DataType,
                                                A0B0B1DataType,
                                                decltype(a0_grid_desc_ak0_m_ak1),
                                                decltype(a0_block_desc_ak0_m_ak1),
                                                A0BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                A0BlockTransferSrcVectorDim,
                                                2,
                                                A0BlockTransferSrcScalarPerVector,
                                                A0BlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemm0KPrefetchStage>(
                a0_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a0_element_op,
                a0_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // B0 matrix blockwise copy
        auto b0_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                B0ElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B0K0PerBlock, Gemm0NPerBlock, B0K1>,
                                                B0BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B0BlockTransferThreadClusterArrangeOrder,
                                                A0B0B1DataType,
                                                A0B0B1DataType,
                                                decltype(b0_grid_desc_bk0_n_bk1),
                                                decltype(b0_block_desc_bk0_n_bk1),
                                                B0BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B0BlockTransferSrcVectorDim,
                                                2,
                                                B0BlockTransferSrcScalarPerVector,
                                                B0BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemm0KPrefetchStage>(
                b0_grid_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                b0_element_op,
                b0_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // Fused Gemm+Gemm pipeline
        // for n in N0:
        //   for k in K0:
        //     acc[m][n] += A[m][k] * B0[k][n]
        //   acc1[m][o] += acc[m][n] * B1[n][o]

        // sanity check
        constexpr index_t KPack = math::max(
            math::lcm(A0K1, B0K1),
            MfmaSelector<A0B0B1DataType, Gemm0MPerXdl, Gemm0NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm0 = BlockwiseGemmXdlops_v2<
            BlockSize,
            A0B0B1DataType,
            Acc0DataType,
            decltype(a0_block_desc_ak0_m_ak1),
            decltype(b0_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a0_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b0_block_desc_bk0_n_bk1)),
            Gemm0MPerBlock,
            Gemm0NPerBlock,
            Gemm0KPerBlock,
            Gemm0MPerXdl,
            Gemm0NPerXdl,
            Gemm0MXdlPerWave,
            Gemm0NXdlPerWave,
            KPack,
            true>{}; // TransposeC

        auto acc0_thread_buf = blockwise_gemm0.GetCThreadBuffer();

        // LDS allocation for A0 and B0: be careful of alignment
        auto a0_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<A0B0B1DataType*>(p_shared) + SharedMemTrait::a0_block_space_offset,
            a0_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b0_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<A0B0B1DataType*>(p_shared) + SharedMemTrait::b0_block_space_offset,
            b0_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a0_block_slice_copy_step = make_multi_index(Gemm0KPerBlock / A0K1, 0, 0);
        constexpr auto b0_block_slice_copy_step = make_multi_index(Gemm0KPerBlock / B0K1, 0, 0);
        const auto a0_block_reset_copy_step =
            make_multi_index(-a0_grid_desc_ak0_m_ak1.GetLength(I0), 0, 0);
        const auto b0_block_reset_copy_step =
            make_multi_index(-b0_grid_desc_bk0_n_bk1.GetLength(I0), Gemm0NPerBlock, 0);

        // gridwise GEMM pipeline
        // Only supports LoopScheduler::Default
        const auto gridwise_gemm0_pipeline =
            GridwiseGemmPipeline_v1_Selector<NumGemm0KPrefetchStage, LoopScheduler::Default>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a0_grid_desc_ak0_m_ak1.GetLength(I0) * a0_grid_desc_ak0_m_ak1.GetLength(I2)) /
            Gemm0KPerBlock);

        //
        // set up Gemm1
        //

        // Acc0 matrix threadwise copy: AccVGPR to VGPR and downcast to XDL input data type
        constexpr auto acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm0.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto m0 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto n0 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto m1 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto n1 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto m2 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto n2 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto n3 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto n4 = acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto b1_block_slice_copy_step = make_multi_index(Gemm1KPerBlock / B1K1, 0, 0);

        // d0 matrix threadwise copy
        constexpr auto d0_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockID
                                                           I1,   // MRepeat
                                                           I1,   // NRepeat
                                                           I1,   // MWaveId
                                                           I1,   // NWaveId
                                                           I1,   // MPerXdl
                                                           I1,   // NGroupNum
                                                           I1,   // NInputNum
                                                           n4)); // registerNum

        auto d0s_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<
                    AddressSpaceEnum::Vgpr,
                    A0B0B1DataType,
                    d0_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize(),
                    true>{};
            },
            Number<NumD0Tensor>{});

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        constexpr auto acc0_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<Gemm0MXdlPerWave>{}, Number<Gemm0NXdlPerWave>{}, n2, n4));

        auto d0s_threadwise_copy = generate_tuple(
            [&](auto i) {
                return ThreadwiseTensorSliceTransfer_v2<
                    A0B0B1DataType,
                    A0B0B1DataType,
                    decltype(d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i]),
                    decltype(d0_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
                    Sequence<I1, I1, I1, I1, I1, I1, I1, I1, I1, n4>,
                    Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
                    9,
                    n4,
                    1,
                    false>(d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                           make_multi_index(block_work_idx[I0], // MBlockId
                                            0,                  // NBlockId
                                            0,                  // mrepeat
                                            0,                  // nrepeat
                                            wave_id[I0],        // MWaveId
                                            wave_id[I1],        // NWaveId
                                            wave_m_n_id[I1],    // MPerXdl
                                            0,                  // group
                                            wave_m_n_id[I0],    // NInputIndex
                                            0));                // register number
            },
            Number<NumD0Tensor>{});
        // acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 to acc0_thread_desc_k0_m_k1
        // n0_n1_n2_n3 -> k0
        // m0_m1_m2 -> m
        // n4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        constexpr auto acc0_thread_desc_k0_m_k1 = transform_tensor_descriptor(
            acc0_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                       make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2)),
                       make_pass_through_transform(n4)),
            make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // A1 matrix in AccVGPR
        // N2 num_groups_per_blk, N3 num_input_blks, N4 group_size
        constexpr auto Acc0N3 =
            blockwise_gemm0.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLength(I6);

        constexpr auto A1ThreadSlice_K0_M_K1 = make_tuple(
            Number<Gemm1KPerBlock / n4 / Acc0N3>{}, Number<m0 * m1 * m2>{}, Number<n4>{});

        constexpr auto A1ThreadSliceK0        = A1ThreadSlice_K0_M_K1[I0];
        constexpr auto A1ThreadSliceM         = A1ThreadSlice_K0_M_K1[I1];
        constexpr auto A1ThreadSliceK1        = A1ThreadSlice_K0_M_K1[I2];
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor(
            A1ThreadSlice_K0_M_K1,
            make_tuple(A1ThreadSliceM * A1ThreadSliceK1, A1ThreadSliceK1, I1));

        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_bk0_n_bk1 = GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            Acc0DataType,
            A0B0B1DataType,
            decltype(acc0_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4>{tensor_operation::element_wise::PassThrough{}};

        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                B0ElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0PerBlock, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                A0B0B1DataType,
                                                A0B0B1DataType,
                                                decltype(b1_grid_desc_bk0_n_bk1),
                                                decltype(b1_block_desc_bk0_n_bk1),
                                                B1BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B1BlockTransferSrcVectorDim,
                                                2,
                                                B1BlockTransferSrcScalarPerVector,
                                                B1BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                B1ThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                1>(b1_grid_desc_bk0_n_bk1,
                                                   make_multi_index(0, n_block_data_idx_on_grid, 0),
                                                   b1_element_op,
                                                   b1_block_desc_bk0_n_bk1,
                                                   make_multi_index(0, 0, 0),
                                                   tensor_operation::element_wise::PassThrough{});

        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, A0B0B1DataType>(
            a1_thread_desc_k0_m_k1.GetElementSpaceSize());

        // reuse LDS space for gemm0's b0_block_buf
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<A0B0B1DataType*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr index_t Gemm1KPack = math::max(
            math::lcm(
                MfmaSelector<A0B0B1DataType, Gemm0MPerXdl, Gemm0NPerXdl>::selected_mfma.group_size,
                B1K1),
            MfmaSelector<A0B0B1DataType, Gemm0MPerXdl, Gemm0NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm1 = BlockwiseGemmXdlops_v2<
            BlockSize,
            A0B0B1DataType,
            Acc1DataType,
            decltype(a1_thread_desc_k0_m_k1),
            decltype(b1_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a1_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b1_block_desc_bk0_n_bk1)),
            Gemm0MPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            Gemm0MPerXdl,
            Gemm0NPerXdl,
            Gemm0MXdlPerWave,
            Gemm1NXdlPerWave,
            Gemm1KPack,
            false,      // TransposeC
            Gemm1KPack, // AMmaKStride
            Gemm1KPack * XdlopsGemm<A0B0B1DataType, Gemm0MPerXdl, Gemm0NPerXdl, Gemm1KPack, false>{}
                             .K0PerXdlops>{                         // BMmaKStride
                                           make_tuple(0, 0, 0, 0)}; // A_origin

        auto c1_thread_buf = blockwise_gemm1.GetCThreadBuffer();

        const index_t num_gemm1_k_block_outer_loop =
            b0_grid_desc_bk0_n_bk1.GetLength(I1) / Gemm0NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = Gemm0NPerBlock / Gemm1KPerBlock;

        // Initialize C1
        c1_thread_buf.Clear();

        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
        do
        {
            // gemm0
            gridwise_gemm0_pipeline.template Run<HasMainKBlockLoop>(a0_grid_desc_ak0_m_ak1,
                                                                    a0_block_desc_ak0_m_ak1,
                                                                    a0_blockwise_copy,
                                                                    a0_grid_buf,
                                                                    a0_block_buf,
                                                                    a0_block_slice_copy_step,
                                                                    b0_grid_desc_bk0_n_bk1,
                                                                    b0_block_desc_bk0_n_bk1,
                                                                    b0_blockwise_copy,
                                                                    b0_grid_buf,
                                                                    b0_block_buf,
                                                                    b0_block_slice_copy_step,
                                                                    blockwise_gemm0,
                                                                    acc0_thread_buf,
                                                                    num_k_block_main_loop);
            // bias+gelu
            {
                static_for<0, Gemm0MXdlPerWave, 1>{}([&](auto mr) {
                    static_for<0, Gemm0NXdlPerWave, 1>{}([&](auto nr) {
                        static_for<0, n2, 1>{}([&](auto groupid) {
                            static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                                d0s_threadwise_copy(i).Run(
                                    d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                                    d0s_grid_buf[i],
                                    d0_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                                    make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                    d0s_thread_buf(i));
                            });

                            static_for<0, n4, 1>{}([&](auto i) {
                                constexpr index_t c_offset = acc0_thread_desc.CalculateOffset(
                                    make_tuple(mr, nr, groupid, i));

                                // get reference to src data
                                const auto src_data_refs = generate_tie(
                                    // return type should be lvalue
                                    [&](auto iSrc) -> const auto& {
                                        return d0s_thread_buf[iSrc][i];
                                    },
                                    Number<NumD0Tensor>{});

                                // get reference to dst data
                                auto dst_data_refs = generate_tie(
                                    // return type should be lvalue
                                    [&](auto) -> auto& {
                                        return acc0_thread_buf(Number<c_offset>{});
                                    },
                                    Number<2>{});

                                unpack2(cde0_element_op, dst_data_refs, src_data_refs);
                            });
                            static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                                d0s_threadwise_copy(i).MoveSrcSliceWindow(
                                    d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                                    make_multi_index(0, 0, 0, 0, 0, 0, 0, 1, 0, 0));
                            });
                        });
                        static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                            d0s_threadwise_copy(i).MoveSrcSliceWindow(
                                d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                                make_multi_index(0, 0, 0, 1, 0, 0, 0, -n2.value, 0, 0));
                        });
                    });
                    static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                        d0s_threadwise_copy(i).MoveSrcSliceWindow(
                            d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                            make_multi_index(0, 0, 1, -Gemm0NXdlPerWave, 0, 0, 0, 0, 0, 0));
                    });
                });
                static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                    d0s_threadwise_copy(i).MoveSrcSliceWindow(
                        d0s_griddesc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5[i],
                        make_multi_index(0, 1, -Gemm0MXdlPerWave, 0, 0, 0, 0, 0, 0, 0));
                });
            }
            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // preload data into LDS
                b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for gemm0 LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        a1_blockwise_copy.Run(acc0_thread_desc_k0_m_k1,
                                              make_tuple(Number<i * A1ThreadSliceK0>{}, I0, I0),
                                              acc0_thread_buf,
                                              a1_thread_desc_k0_m_k1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                        block_sync_lds();

                        blockwise_gemm1.Run(a1_thread_buf, b1_block_buf, c1_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);
                    });
                }
                // tail
                {
                    a1_blockwise_copy.Run(
                        acc0_thread_desc_k0_m_k1,
                        make_tuple(
                            Number<(num_gemm1_k_block_inner_loop - 1) * A1ThreadSliceK0>{}, I0, I0),
                        acc0_thread_buf,
                        a1_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    block_sync_lds();

                    blockwise_gemm1.Run(a1_thread_buf, b1_block_buf, c1_thread_buf);
                }
            } // end gemm1

            a0_blockwise_copy.MoveSrcSliceWindow(a0_grid_desc_ak0_m_ak1,
                                                 a0_block_reset_copy_step); // rewind K
            b0_blockwise_copy.MoveSrcSliceWindow(b0_grid_desc_bk0_n_bk1,
                                                 b0_block_reset_copy_step); // rewind K and step N

            block_sync_lds(); // wait for gemm1 LDS read
        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop

        // shuffle C1 and write out
        {
            static_assert(Gemm0MXdlPerWave % C1ShuffleGemm0MXdlPerWavePerShuffle == 0 &&
                              Gemm1NXdlPerWave % C1ShuffleGemm0NXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = Gemm0MPerBlock / (Gemm0MXdlPerWave * Gemm0MPerXdl);
            constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * Gemm0NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c1_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm1.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm1.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
            constexpr auto N0 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
            constexpr auto M1 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
            constexpr auto N1 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
            constexpr auto M2 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
            constexpr auto M3 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
            constexpr auto M4 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
            constexpr auto N2 = c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

            constexpr auto c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetC1ShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c1_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<C1ShuffleDataType*>(p_shared),
                c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<C1ShuffleGemm0MXdlPerWavePerShuffle>{}, // M0 (Gemm0MXdlPerWave) per
                                                                       // shuffle
                        M1,                                            // M1 = MWave
                        M2, // M2 * M3 * M4 = Gemm0MPerXdl
                        M3,
                        M4)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<C1ShuffleGemm0NXdlPerWavePerShuffle>{}, // N0 (Gemm0NXdlPerWave) per
                                                                       // shuffle
                        N1,                                            // N1 = NWave
                        N2))),                                         // N2 = Gemm0NPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM C1 matrix starting index
            const auto c1_thread_mtx_on_block =
                blockwise_gemm1.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c1_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c1_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c1_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<Acc1DataType,
                                                   C1ShuffleDataType,
                                                   decltype(c1_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   tensor_operation::element_wise::PassThrough,
                                                   Sequence<C1ShuffleGemm0MXdlPerWavePerShuffle,
                                                            C1ShuffleGemm0NXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    tensor_operation::element_wise::PassThrough{}};

            // tuple of reference to C/Ds tensor descriptors
            const auto c1_d1s_desc_refs = concat_tuple_of_reference(
                tie(c1_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                generate_tie(
                    [&](auto i) -> const auto& // return type should be reference
                    { return d1s_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                    Number<NumD1Tensor>{}));

            // tuple of reference to C/Ds tensor descriptors
            const auto c1_d1s_buf_refs = concat_tuple_of_reference(
                tie(c1_shuffle_block_buf),
                generate_tie(
                    [&](auto i) -> const auto& // return type should be reference
                    { return d1s_grid_buf[i]; },
                    Number<NumD1Tensor>{}));

            // tuple of starting index of C/Ds blockwise copy
            const auto idx_c1_d1s_block_begin = container_concat(
                make_tuple(make_multi_index(0, 0, 0, 0)),
                generate_tuple(
                    [&](auto) {
                        return make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0);
                    },
                    Number<NumD1Tensor>{}));

            // shuffle: blockwise copy C from LDS to global
            auto cde1_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v7<
                ThisThreadBlock,
                decltype(container_concat(make_tuple(C1ShuffleDataType{}), D1sDataType{})),
                Tuple<E1DataType>,
                decltype(c1_d1s_desc_refs),
                decltype(tie(e1_grid_desc_mblock_mperblock_nblock_nperblock)),
                CDE1ElementwiseOperation,
                Sequence<static_cast<index_t>(E1GlobalMemoryDataOperation)>, // FIXME: make Sequence
                                                                             // support arbitray
                                                                             // type
                Sequence<1,
                         C1ShuffleGemm0MXdlPerWavePerShuffle * MWave * Gemm0MPerXdl,
                         1,
                         C1ShuffleGemm0NXdlPerWavePerShuffle * NWave *
                             Gemm0NPerXdl>, // BlockSliceLengths,
                CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                Sequence<0, 1, 2, 3>, // typename DimAccessOrder,
                3,                    // index_t VectorDim,
                CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
                sequence_merge_t<
                    Sequence<true>,
                    uniform_sequence_gen_t<NumD1Tensor,
                                           false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                Sequence<false>>                    // ThreadTransferDstResetCoordinateAfterRunFlags
                {c1_d1s_desc_refs,
                 idx_c1_d1s_block_begin,
                 tie(e1_grid_desc_mblock_mperblock_nblock_nperblock),
                 make_tuple(make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0)),
                 cde1_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c1_vgpr =
                SpaceFillingCurve<Sequence<Gemm0MXdlPerWave, Gemm1NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<C1ShuffleGemm0MXdlPerWavePerShuffle,
                                           C1ShuffleGemm0NXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           M2,
                                           1,
                                           M4,
                                           1>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_e1_global = SpaceFillingCurve<
                Sequence<1, Gemm0MPerBlock, 1, Gemm1NPerBlock>,
                Sequence<0, 2, 1, 3>,
                Sequence<1,
                         C1ShuffleGemm0MXdlPerWavePerShuffle * MWave * Gemm0MPerXdl,
                         1,
                         C1ShuffleGemm0NXdlPerWavePerShuffle * NWave * Gemm0NPerXdl>>{};

            constexpr index_t num_access = sfc_c1_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_e1_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c1_thread_copy_vgpr_to_lds.Run(c1_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                               sfc_c1_vgpr.GetIndexTupleOfNumber(access_id),
                                               c1_thread_buf,
                                               c1_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                               c1_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                cde1_shuffle_block_copy_lds_to_global.Run(
                    c1_d1s_desc_refs,
                    c1_d1s_buf_refs,
                    tie(e1_grid_desc_mblock_mperblock_nblock_nperblock),
                    tie(e1_grid_buf));

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto e1_global_step = sfc_e1_global.GetForwardStep(access_id);

                    // move on D1s
                    static_for<0, NumD1Tensor, 1>{}([&](auto i) {
                        cde1_shuffle_block_copy_lds_to_global.MoveSrcSliceWindow(
                            c1_d1s_desc_refs, i + I1, e1_global_step);
                    });

                    // move on C
                    cde1_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        tie(e1_grid_desc_mblock_mperblock_nblock_nperblock), I0, e1_global_step);
                }
            });
        }
    }
};

} // namespace ck
