// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
template <typename FloatAB,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename DsDataType,
          typename FloatE,
          typename FloatReduceAcc,
          typename RsDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation,
          typename ThreadReduceOperations,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          typename RsGlobalMemoryDataOperation,
          typename AGridDesc_M_K,
          typename BGridDesc_N_K,
          typename EGridDesc_M_N,
          typename RGridDesc_M,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDRThreadTransferClusterLengths_MPerBlock_NPerBlock,
          index_t CDEReduceThreadTransferScalarPerVector_NPerBlock,
          index_t RThreadTransferDstScalarPerVector_MPerBlock,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseGemmMultipleDMultipleR_k0mk1_k0nk1_mn_xdl_cshuffle_v1
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr index_t NumRTensor = RsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0PerBlock, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0PerBlock, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    // ck::Tuple<const T0DataType*, const T1DataType*, ...>
    template <typename Ts, bool isConst = true>
    static constexpr auto MakeTsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using T = remove_cvref_t<tuple_element_t<i.value, Ts>>;
                if constexpr(isConst)
                    return static_cast<const T*>(nullptr);
                else
                    return static_cast<T*>(nullptr);
            },
            Number<Ts::Size()>{});
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned + b_block_space_size_aligned) *
                             sizeof(FloatAB),
                         c_block_size * sizeof(FloatCShuffle));
    }

    // A desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // B desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2ETileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc_M_K& a_grid_desc_m_k,
                                                            const BGridDesc_N_K& b_grid_desc_n_k,
                                                            const EGridDesc_M_N& e_grid_desc_m_n,
                                                            const RGridDesc_M& r_grid_desc_m,
                                                            const Block2ETileMap& block_2_etile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        static_assert(AGridDesc_M_K::GetNumOfDimension() == 2);
        static_assert(BGridDesc_N_K::GetNumOfDimension() == 2);
        static_assert(EGridDesc_M_N::GetNumOfDimension() == 2);

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
            return false;

        if(M != r_grid_desc_m.GetLength(I0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        if(!block_2_etile_map.CheckValidity(e_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto
    MakeRGridDescriptor_MBlock_MPerBlock(const RGridDesc_M& r_grid_desc_m)
    {
        const auto M      = r_grid_desc_m.GetLength(I0);
        const auto MBlock = M / MPerBlock;

        const auto r_grid_desc_mblock_mperblock = transform_tensor_descriptor(
            r_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1>{}));

        return r_grid_desc_mblock_mperblock;
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n);
    }

    using DefaultAGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using DefaultBGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;
    using EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}))>;

    // Support 2 dimension in the future. Not only M
    using RGridDescriptor_MBlock_MPerBlock =
        remove_cvref_t<decltype(MakeRGridDescriptor_MBlock_MPerBlock(RGridDesc_M{}))>;

    using DefaultBlock2ETileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    using DsGridPointer = decltype(MakeTsGridPointer<DsDataType, true>());
    using RsGridPointer = decltype(MakeTsGridPointer<RsDataType, false>());

    template <bool HasMainKBlockLoop,
              typename AGridDesc_AK0_M_AK1,
              typename BGridDesc_BK0_N_BK1,
              typename Block2ETileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        DsGridPointer p_ds_grid,
        FloatE* __restrict__ p_e_grid,
        RsGridPointer p_rs_grid,
        void* __restrict__ p_shared,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op,
        const QsElementwiseOperation& qs_element_op,
        const RsElementwiseOperation& rs_element_op,
        const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
        const StaticallyIndexedArray<EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                                     NumDTensor>&
            ds_grid_desc_mblock_mperblock_nblock_nperblock, // FIXME: Ds desc may be of different
        const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
            e_grid_desc_mblock_mperblock_nblock_nperblock,
        const StaticallyIndexedArray<RGridDescriptor_MBlock_MPerBlock,
                                     NumRTensor>&
            rs_grid_desc_mblock_mperblock, // FIXME: Rs desc may be of different
        const Block2ETileMap& block_2_etile_map)
    {
        // FIXME - Share code with other gemm kernel
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        auto rs_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_rs_grid(i), rs_grid_desc_mblock_mperblock[i].GetElementSpaceSize());
            },
            Number<NumRTensor>{});

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_etile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            FloatAB,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
                                                               a_block_desc_ak0_m_ak1,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               b_grid_desc_bk0_n_bk1,
                                                               b_block_desc_bk0_n_bk1,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // shuffle C + Ds + reduction + write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatCShuffle*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2,                                      // M2 * M3 * M4 = MPerXdl
                        M3,
                        M4)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2))),                                   // N2 = NPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

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
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
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
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           M2,
                                           1,
                                           M4,
                                           1>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_der_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            // TODO: this should be implemented as a blockwise reduction
            // LDS c_reduce_block_desc_mperblock_nperblock
            constexpr auto c_reduce_block_desc_mperblock_nperblock = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_pass_through_transform(
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I1)),
                    make_freeze_transform(I0),
                    make_pass_through_transform(
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I3))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<>{}, Sequence<1>{}));

            static_assert(CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I0) *
                                  CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I1) ==
                              BlockSize,
                          "wrong!");

            static_assert((CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl) %
                                      CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I0) ==
                                  0 &&
                              (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl) %
                                      CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I1) ==
                                  0,
                          "wrong!");

            constexpr index_t mreduce_per_thread =
                (CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl) /
                CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I0);

            constexpr index_t nreduce_per_thread =
                (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl) /
                CDRThreadTransferClusterLengths_MPerBlock_NPerBlock::At(I1);

            constexpr auto c_reduce_thread_lengths_mperblock_nperblock =
                Sequence<mreduce_per_thread, nreduce_per_thread>{};

            // VGPR cde_reduce_thread_desc_mperblock_nperblock
            constexpr auto cde_reduce_thread_desc_mperblock_nperblock =
                make_naive_tensor_descriptor_packed(
                    make_tuple(Number<mreduce_per_thread>{}, Number<nreduce_per_thread>{}));

            constexpr auto r_thread_desc_mperblock =
                make_naive_tensor_descriptor_packed(make_tuple(Number<mreduce_per_thread>{}));

            constexpr auto r_thread_desc_mblock_mperblock =
                make_naive_tensor_descriptor_packed(make_tuple(I1, Number<mreduce_per_thread>{}));

            auto e_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatReduceAcc>(
                cde_reduce_thread_desc_mperblock_nperblock.GetElementSpaceSize());

            // reduce: threadwise copy from LDS to VGPR
            constexpr auto c_reduce_thread_cluster_desc = make_cluster_descriptor(
                CDRThreadTransferClusterLengths_MPerBlock_NPerBlock{}, Sequence<1, 0>{});

            const auto c_reduce_thread_cluster_idx =
                c_reduce_thread_cluster_desc.CalculateBottomIndex(
                    make_multi_index(get_thread_local_1d_id()));

            const auto c_reduce_thread_data_idx_begin =
                c_reduce_thread_cluster_idx * c_reduce_thread_lengths_mperblock_nperblock;

            // To apply D0, D1, ... and reduction.
            // Copy c shuffle from LDS back to VGPR
            auto c_reduce_thread_copy_lds_to_vgpr = ThreadwiseTensorSliceTransfer_v2<
                FloatCShuffle,
                FloatReduceAcc,
                decltype(c_reduce_block_desc_mperblock_nperblock),
                decltype(cde_reduce_thread_desc_mperblock_nperblock),
                decltype(c_reduce_thread_lengths_mperblock_nperblock),
                Sequence<0, 1>,
                1,
                CDEReduceThreadTransferScalarPerVector_NPerBlock,
                1,
                true>{c_reduce_block_desc_mperblock_nperblock, c_reduce_thread_data_idx_begin};

            // Copy result of reduction back from VGPR to global
            auto reduce_tuple_thread_copy_vgpr_to_global = generate_tuple(
                [&](auto I) {
                    auto p_r_grid                     = p_rs_grid[I];
                    auto r_element_op                 = rs_element_op[I];
                    auto r_grid_desc_mblock_mperblock = rs_grid_desc_mblock_mperblock[I];

                    return ThreadwiseTensorSliceTransfer_v1r3<
                        FloatReduceAcc,
                        remove_pointer_t<decltype(p_r_grid)>,
                        decltype(r_thread_desc_mblock_mperblock),
                        decltype(r_grid_desc_mblock_mperblock),
                        decltype(r_element_op),
                        Sequence<1, mreduce_per_thread>,
                        Sequence<0, 1>,
                        1,
                        RThreadTransferDstScalarPerVector_MPerBlock,
                        RsGlobalMemoryDataOperation::At(I),
                        1,
                        false>{r_grid_desc_mblock_mperblock,
                               make_multi_index(block_work_idx[I0],                  // mblock
                                                c_reduce_thread_data_idx_begin[I0]), // mperblock
                               r_element_op};
                },
                Number<NumRTensor>{});

            // D0, D1, ..., Dn
            constexpr auto cde_reduce_thread_desc_I1_mperblock_I1_nperblock =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1, Number<mreduce_per_thread>{}, I1, Number<nreduce_per_thread>{}));

            // FIXME: Decrease usage of VGPR
            // Apply pointwise lambda function from multi-source (Global and LDS) into VGPR
            auto ds_thread_buf = generate_tuple(
                [&](auto) {
                    return make_static_buffer<AddressSpaceEnum::Vgpr, FloatReduceAcc>(
                        cde_reduce_thread_desc_I1_mperblock_I1_nperblock.GetElementSpaceSize());
                },
                Number<NumDTensor>{});

            // Copy D0, D1, ..., Dn from global to VGPR
            auto ds_thread_copy_global_to_vgpr = generate_tuple(
                [&](auto I) {
                    using DDataType = remove_cvref_t<tuple_element_t<I.value, DsDataType>>;
                    return ThreadwiseTensorSliceTransfer_v2<
                        DDataType,
                        FloatReduceAcc,
                        decltype(ds_grid_desc_mblock_mperblock_nblock_nperblock[I]),
                        decltype(cde_reduce_thread_desc_I1_mperblock_I1_nperblock),
                        Sequence<I1, mreduce_per_thread, I1, nreduce_per_thread>,
                        Sequence<0, 1, 2, 3>,
                        3,
                        CDEReduceThreadTransferScalarPerVector_NPerBlock,
                        1,
                        true>(ds_grid_desc_mblock_mperblock_nblock_nperblock[I],
                              make_multi_index(
                                  I0,
                                  m_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I0],
                                  I0,
                                  n_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I1]));
                },
                Number<NumDTensor>{});

            auto e_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                FloatReduceAcc,
                FloatE,
                decltype(cde_reduce_thread_desc_I1_mperblock_I1_nperblock),
                decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
                tensor_operation::element_wise::PassThrough,
                Sequence<I1, mreduce_per_thread, I1, nreduce_per_thread>, // SliceLengths
                Sequence<0, 1, 2, 3>,                                     // DimAccessOrder
                3,                                                        // DstVectorDim
                CDEReduceThreadTransferScalarPerVector_NPerBlock,
                InMemoryDataOperationEnum::Set,
                1,
                true>{
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                make_multi_index(I0,
                                 m_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I0],
                                 I0,
                                 n_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I1]),
                tensor_operation::element_wise::PassThrough{}};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_der_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to read from LDS
                block_sync_lds();

                // each thread shuffle data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              c_shuffle_block_buf);

                // make sure it's safe to write to LDS
                block_sync_lds();

                // Get shuffle data from LDS to VGPR
                c_reduce_thread_copy_lds_to_vgpr.Run(c_reduce_block_desc_mperblock_nperblock,
                                                     c_shuffle_block_buf,
                                                     cde_reduce_thread_desc_mperblock_nperblock,
                                                     make_tuple(I0, I0),
                                                     e_thread_buf);

                // Global read D0, D1, ...
                static_for<0, NumDTensor, 1>{}([&](auto Id) {
                    auto& d_thread_copy_global_to_vgpr = ds_thread_copy_global_to_vgpr(Id);
                    d_thread_copy_global_to_vgpr.Run(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[Id],
                        ds_grid_buf[Id],
                        cde_reduce_thread_desc_I1_mperblock_I1_nperblock,
                        make_tuple(I0, I0, I0, I0),
                        ds_thread_buf(Id));

                    if constexpr(access_id < num_access - 1)
                    {
                        // move on D0, D1, ...
                        constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                        d_thread_copy_global_to_vgpr.MoveSrcSliceWindow(
                            ds_grid_desc_mblock_mperblock_nblock_nperblock[Id], de_global_step);
                    }
                });

                // cde_element_op(e, c, d0, d1, ...);
                static_for<0, cde_reduce_thread_desc_mperblock_nperblock.GetElementSize(), 1>{}(
                    [&](auto i) {
                        const auto c_ds_src_data_refs = concat_tuple_of_reference(
                            tie(e_thread_buf[i]),
                            generate_tie(
                                [&](auto Id) -> const auto& { return ds_thread_buf[Id][i]; },
                                Number<NumDTensor>{}));
                        auto e_dst_data_refs = tie(e_thread_buf(i));
                        unpack2(cde_element_op, e_dst_data_refs, c_ds_src_data_refs);
                    });

                // Global write E
                e_thread_copy_vgpr_to_global.Run(cde_reduce_thread_desc_I1_mperblock_I1_nperblock,
                                                 make_tuple(I0, I0, I0, I0),
                                                 e_thread_buf,
                                                 e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                 e_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    // move on E
                    constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                    e_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                        e_grid_desc_mblock_mperblock_nblock_nperblock, de_global_step);
                }

                // reduction
                static_for<0, NumRTensor, 1>{}([&](auto Ir) {
                    auto r_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatReduceAcc>(
                        r_thread_desc_mperblock.GetElementSpaceSize());

                    auto& reduce_thread_copy_vgpr_to_global =
                        reduce_tuple_thread_copy_vgpr_to_global(Ir);

                    using ThreadReduceOperation =
                        remove_cvref_t<decltype(ThreadReduceOperations{}[Ir])>;

                    using ThreadwiseReduce =
                        ThreadwiseReduction<FloatReduceAcc,
                                            decltype(cde_reduce_thread_desc_mperblock_nperblock),
                                            decltype(r_thread_desc_mperblock),
                                            ThreadReduceOperation,
                                            false>;

                    // threadwise reduction
                    const auto reduce_identityVal =
                        ThreadReduceOperation::template GetIdentityValue<FloatReduceAcc>();
                    static_for<0, mreduce_per_thread, 1>{}(
                        [&](auto I) { r_thread_buf(I) = reduce_identityVal; });
                    static_for<0, mreduce_per_thread, 1>{}([&](auto im) {
                        static_for<0, nreduce_per_thread, 1>{}([&](auto in) {
                            constexpr auto offset =
                                Number<cde_reduce_thread_desc_mperblock_nperblock.CalculateOffset(
                                    make_tuple(im, in))>{};

                            qs_element_op[Ir](e_thread_buf(offset), e_thread_buf(offset));
                        });
                    });
                    ThreadwiseReduce::Reduce(e_thread_buf, r_thread_buf);

                    // gridwise reduction
                    reduce_thread_copy_vgpr_to_global.Run(r_thread_desc_mblock_mperblock,
                                                          make_tuple(I0, I0),
                                                          r_thread_buf,
                                                          rs_grid_desc_mblock_mperblock[Ir],
                                                          rs_grid_buf(Ir));

                    if constexpr(access_id < num_access - 1)
                    {
                        // move on R0, R1, ...
                        constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                        reduce_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                            rs_grid_desc_mblock_mperblock[Ir],
                            make_tuple(de_global_step[I0], de_global_step[I1]));
                    }
                });
            }); // copy c, d, e + reduction

        } // shuffle C + Ds + reduction + write out
    }     // Run
};

} // namespace ck
