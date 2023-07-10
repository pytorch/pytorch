// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename InputDataType,
          typename OutputDataType,
          typename ZDataType,
          typename GemmDataType,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatLSE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename SElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename QGridDesc_K0_M_K1,
          typename KGridDesc_K0_N_K1,
          typename ZGridDesc_M_N,
          typename VGridDesc_N0_O_N1,
          typename CGridDesc_M_N,
          typename LSEGridDesc_M,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t B1K1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          index_t Gemm2NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t BBlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          index_t B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          bool PadN,
          bool MaskOutUpperTriangle,
          bool Deterministic,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedMultiheadAttentionBackward_Xdl_CShuffle_V2
{
    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

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

    static constexpr auto WaveSize = 64;
    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1KPerBlock / B1K1Value>{};
    static constexpr auto B1K1 = Number<B1K1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    // C desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(const ZGridDesc_M_N& z_grid_desc_m_n)
    {
        const auto M = z_grid_desc_m_n.GetLength(I0);
        const auto N = z_grid_desc_m_n.GetLength(I1);

        constexpr auto mfma = MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto N3   = mfma.num_groups_per_blk;
        constexpr auto N4   = mfma.num_input_blks;
        constexpr auto N5   = mfma.group_size;
        return transform_tensor_descriptor(
            z_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, MPerXdl)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, N3, N4, N5))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7, 8, 9>{}));
    }
    __host__ __device__ static constexpr auto
    MakeZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(const index_t M, const index_t N)
    {
        constexpr auto mfma = MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto N3   = mfma.num_groups_per_blk;
        constexpr auto N4   = mfma.num_input_blks;
        constexpr auto N5   = mfma.group_size;
        return transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(M, N)),
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, MPerXdl)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, N3, N4, N5))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7, 8, 9>{}));
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
            make_tuple(make_merge_transform(make_tuple(WaveSize / MPerXdl, MPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, 1, 1>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    template <typename AccThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4>
    __host__ __device__ static constexpr auto GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
        const AccThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4& acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4)
    {
        // acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 to a_src_thread_desc_k0_m_k1
        // n0_n1_n2_n3 -> k0
        // m0_m1_m2 -> m
        // n4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        const auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        const auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        const auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        const auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        const auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        const auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        const auto n3 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        const auto n4 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        return transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                       make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2)),
                       make_pass_through_transform(n4)),
            make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
    }

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0, Number<Gemm1NPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1NPerBlock + B1BlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    template <typename Gemm2Param>
    __host__ __device__ static constexpr auto GetA2BlockDescriptor_M0_N_M1()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<Gemm2Param::A_M0>{},
                       Number<Gemm2Param::Free0_N>{},
                       Number<Gemm2Param::A_M1>{}),
            make_tuple(Number<Gemm2Param::Free0_N + Gemm2Param::A_LdsPad>{} *
                           Number<Gemm2Param::A_M1>{},
                       Number<Gemm2Param::A_M1>{},
                       I1));
    }

    template <typename Gemm2Param>
    __host__ __device__ static constexpr auto GetB2BlockDescriptor_M0_O_M1()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<Gemm2Param::B_M0>{},
                       Number<Gemm2Param::Free1_O>{},
                       Number<Gemm2Param::B_M1>{}),
            make_tuple(Number<Gemm2Param::Free1_O + Gemm2Param::B_LdsPad>{} *
                           Number<Gemm2Param::B_M1>{},
                       Number<Gemm2Param::B_M1>{},
                       I1));
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
                  const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
                  const VGridDesc_N0_O_N1& v_grid_desc_n0_o_n1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M      = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto N      = k_grid_desc_k0_n_k1.GetLength(I1);
        const auto K      = q_grid_desc_k0_m_k1.GetLength(I0) * q_grid_desc_k0_m_k1.GetLength(I2);
        const auto Gemm1N = v_grid_desc_n0_o_n1.GetLength(I1);

        // This assumption reduces implemention complexity by categorizing 6 separate GEMMs into 3
        // types of GEMM operations, therefore some code body can be reused accordingly
        // P_MNK / dP_MNO Gemm (Gemm0 rcr)
        // Y_MON / dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)
        if(Gemm1N != K)
        {
            std::cerr << "SizeK must be equal to SizeO (equal attention head size)" << '\n';
            return false;
        }

        if(!(M == c_grid_desc_m_n.GetLength(I0) && Gemm1N == c_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0 &&
             Gemm1N % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = NPerBlock / Gemm1KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
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
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / Gemm1NPerBlock;

        const auto y_grid_desc_mblock_mperblock_oblock_operblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return y_grid_desc_mblock_mperblock_oblock_operblock;
    }

    __host__ __device__ static constexpr auto
    MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(const LSEGridDesc_M& lse_grid_desc_m)
    {
        const index_t M         = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock    = M / MPerBlock;
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

        const auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(
                make_tuple(MBlock, Number<MXdlPerWave>{}, MWave, Number<MPerXdl>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        return lse_grid_desc_mblock_mrepeat_mwave_mperxdl;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    using ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5 = remove_cvref_t<decltype(
        MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(ZGridDesc_M_N{}))>;

    // S / dP Gemm (type 1 rcr)
    struct Gemm0
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }

        template <typename GridDesc_K0_M_K1>
        using ABlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_M_K1,
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        template <typename GridDesc_K0_N_K1>
        using BBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        static constexpr index_t KPack =
            math::max(math::lcm(AK1, BK1),
                      MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>; // TransposeC

        static constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        static constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);
    };

    // Y / dQ Gemm (type 2 rrr)
    template <typename ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4,
              typename ASrcBlockDesc_M0_N0_M1_N1_M2_N2_N3_N4>
    struct Gemm1
    {
        private:
        static constexpr auto m0 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I0);
        static constexpr auto n0 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I1);
        static constexpr auto m1 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I2);
        static constexpr auto n1 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I3);
        static constexpr auto m2 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I4);
        static constexpr auto n2 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I5);
        static constexpr auto n3 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I6);
        static constexpr auto n4 = ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I7);

        // N2 num_groups_per_blk, N3 num_input_blks, N4 group_size
        static constexpr auto N3 = ASrcBlockDesc_M0_N0_M1_N1_M2_N2_N3_N4{}.GetLength(I6);

        public:
        static constexpr auto AThreadSliceLength_K0 = Number<Gemm1KPerBlock / n4 / N3>{};
        static constexpr auto AThreadSliceLength_M  = Number<m0 * m1 * m2>{};
        static constexpr auto AThreadSliceLength_K1 = Number<n4>{};

        // A source matrix layout in AccVGPR
        static constexpr auto a_src_thread_desc_k0_m_k1 =
            GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
                ASrcThreadDesc_M0_N0_M1_N1_M2_N2_N3_N4{});

        // A matrix in VGPR memory, dst of AccVGPR-to-VGPR copy
        static constexpr auto a_thread_desc_k0_m_k1 = make_naive_tensor_descriptor_packed(
            make_tuple(AThreadSliceLength_K0, AThreadSliceLength_M, AThreadSliceLength_K1));

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        static constexpr auto ASrcScalarPerVector = n4;

        using AThreadSliceLengths_K0_M_K1 = decltype(a_thread_desc_k0_m_k1.GetLengths());

        using ABlockwiseCopy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatGemmAcc,
            GemmDataType,
            decltype(a_src_thread_desc_k0_m_k1),
            decltype(a_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            AThreadSliceLengths_K0_M_K1,
            Sequence<1, 0, 2>,
            2,
            ASrcScalarPerVector>;

        template <typename GridDesc_K0_N_K1>
        using BBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(b_block_desc_bk0_n_bk1),
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
                                                NumGemmKPrefetchStage>;

        // for a_block_slice_copy_step to be able to address static buffers, it MUST be a
        // tuple-based container as well as containing ONLY integral constants
        static constexpr auto a_block_slice_copy_step = make_tuple(AThreadSliceLength_K0, I0, I0);
        static constexpr auto b_block_slice_copy_step =
            make_multi_index(Gemm1KPerBlock / B1K1, 0, 0);

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
        static constexpr index_t GemmKPack =
            MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.group_size;

        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_thread_desc_k0_m_k1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            Gemm1NXdlPerWave,
            GemmKPack,
            true,      // TransposeC
            GemmKPack, // AMmaKStride
            GemmKPack * XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, GemmKPack, false>{}
                            .K0PerXdlops /* BMmaKStride */>;
    };

    // dV / dK Gemm (type 3 crr)
    // Describes tuning parameter for C2_n_o = A2_n_m * B2_m_o
    template <index_t Sum_M_ = MPerXdl * 2>
    struct Gemm2Params_N_O_M_
    {
        static constexpr index_t Free0_N = NPerBlock;
        static constexpr index_t Free1_O = Gemm1NPerBlock;
        static constexpr index_t Sum_M   = Sum_M_;

        static constexpr index_t A_M1     = 8; // P will be row-major
        static constexpr index_t A_M0     = Sum_M / A_M1;
        static constexpr index_t A_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static constexpr index_t B_M1     = 2; // dY assumed row-major, typically =2 for fp16
        static constexpr index_t B_M0     = Sum_M / B_M1;
        static constexpr index_t B_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static_assert(Sum_M % MPerXdl == 0, "");

        static constexpr index_t BSrcVectorDim       = 1; // Free1_O dimension
        static constexpr index_t BSrcScalarPerVector = 4;

        static constexpr index_t GemmNWave   = Free0_N / Gemm2NXdlPerWave / MPerXdl;
        static constexpr index_t GemmOWave   = BlockSize / get_warp_size() / GemmNWave;
        static constexpr index_t GemmNRepeat = Gemm2NXdlPerWave;
        static constexpr index_t GemmORepeat = Free1_O / GemmOWave / NPerXdl;
        static constexpr index_t GemmMPack =
            math::max(math::lcm(A_M1, B_M1),
                      MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        using BBlockSliceLengths = Sequence<B_M0, Free1_O, B_M1>;
        using BThreadClusterLengths =
            Sequence<BlockSize / (Free1_O / BSrcScalarPerVector), Free1_O / BSrcScalarPerVector, 1>;
        using BThreadClusterArrangeOrder = Sequence<0, 2, 1>;

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_N0_M1_N1_M2_N2()
        {
            // perform manual unmerge: m -> m_repeat, m_waves, m_per_xdl
            constexpr index_t m  = Gemm2Params_N_O_M::Sum_M - 1;
            constexpr index_t m2 = m % MPerXdl;
            constexpr index_t m1 = m / MPerXdl % Gemm0MWaves;
            constexpr index_t m0 = m / MPerXdl / Gemm0MWaves % MXdlPerWave;

            // perform manual unmerge: n -> n_repeat, n_waves, n_per_xdl
            constexpr index_t n  = Gemm2Params_N_O_M::Free0_N - 1;
            constexpr index_t n2 = n % NPerXdl;
            constexpr index_t n1 = n / NPerXdl % Gemm0NWaves;
            constexpr index_t n0 = n / NPerXdl / Gemm0NWaves % NXdlPerWave;

            // assume 256 decomposed into 2 x 4 x 32
            // 1d idx ( 32 - 1) -> 3d idx 0, 0, 31 -> 3d dim 1 x 1 x 32
            // 1d idx (256 - 1) -> 3d idx 1, 3, 31 -> 3d dim 2 x 4 x 32
            return Sequence<m0, n0, m1, n1, m2, n2>{} + Sequence<1, 1, 1, 1, 1, 1>{};
        }

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_N0_M1_N1()
        {
            return generate_sequence_v2(
                [](auto I) { return GetABlockSliceLengths_M0_N0_M1_N1_M2_N2().At(I); },
                Number<4>{});
        }

        using ABlockSliceLengths_M0_N0_M1_N1 = decltype(GetABlockSliceLengths_M0_N0_M1_N1());
    };
    using Gemm2Params_N_O_M = Gemm2Params_N_O_M_<>; // tune later

    // dV / dK Gemm (type 3 crr)
    template <typename Gemm2Params_N_O_M, typename ASrcBlockwiseGemm>
    struct Gemm2
    {
        private:
        static constexpr auto a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            ASrcBlockwiseGemm::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
        static constexpr auto M0 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0); // repeat
        static constexpr auto N0 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        static constexpr auto M1 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2); // wave
        static constexpr auto N1 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        static constexpr auto M2 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4); // xdl
        static constexpr auto N2 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        static constexpr auto N3 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        static constexpr auto N4 = a_src_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        public:
        // A source matrix layout in VGPR, src of VGPR-to-LDS copy
        static constexpr auto a_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            ASrcBlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_m0_n_m1 =
            GetA2BlockDescriptor_M0_N_M1<Gemm2Params_N_O_M>();

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_m0_o_m1 =
            GetB2BlockDescriptor_M0_O_M1<Gemm2Params_N_O_M>();

        __host__ __device__ static constexpr auto MakeABlockDesc_M0_N0_M1_N1_M2_N2_N3_N4()
        {
            const auto M0_ = a_block_desc_m0_n_m1.GetLength(I0);
            const auto N_  = a_block_desc_m0_n_m1.GetLength(I1);
            const auto M1_ = a_block_desc_m0_n_m1.GetLength(I2);

            const auto a_block_desc_m_n = transform_tensor_descriptor(
                a_block_desc_m0_n_m1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(M0_, M1_)),
                           make_pass_through_transform(N_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            return transform_tensor_descriptor(
                a_block_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(I1, M1, M2)),
                           make_unmerge_transform(make_tuple(I1, N1, N2, N3, N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
        }

        // Note: we will perform sub-workgroup VGPR-to-LDS copy to save LDS space, therefore the
        // destination coordinate can overlap between wavefronts in a workgroup as seen in the mod
        // operation before returning the values
        __host__ __device__ static auto MakeAThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4()
        {
            const auto a_thread_origin_on_block_idx =
                ASrcBlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0);

            constexpr auto c_block_slice_lengths_m0_n0_m1_n1 =
                typename Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1{}; // mrepeat, nrepeat,
                                                                              // mwaves, nwaves,

            return make_tuple(
                a_thread_origin_on_block_idx[I0],                                         // mrepeat
                a_thread_origin_on_block_idx[I1],                                         // nrepeat
                a_thread_origin_on_block_idx[I2] % c_block_slice_lengths_m0_n0_m1_n1[I2], // mwave
                a_thread_origin_on_block_idx[I3] % c_block_slice_lengths_m0_n0_m1_n1[I3], // nwave
                a_thread_origin_on_block_idx[I4],                                         // xdlops
                a_thread_origin_on_block_idx[I5],
                a_thread_origin_on_block_idx[I6],
                a_thread_origin_on_block_idx[I7]);
        }

        static constexpr auto a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            MakeABlockDesc_M0_N0_M1_N1_M2_N2_N3_N4();

        using ASrcBlockSliceWindowIterator =
            SpaceFillingCurve<Sequence<M0, N0, M1, N1>,
                              Sequence<0, 1, 2, 3>,
                              typename Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1,
                              false>;

        template <typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using ABlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            GemmDataType,
            decltype(a_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            decltype(a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            ElementwiseOp,
            Sequence<Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(
                         I0), // ThreadSliceLengths
                     Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(I1),
                     I1,
                     I1,
                     I1,
                     N2,
                     I1,
                     N4>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>;

        template <typename GridDesc_M0_O_M1>
        using BBlockwiseCopy = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            tensor_operation::element_wise::PassThrough,
            tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            typename Gemm2Params_N_O_M::BBlockSliceLengths,
            typename Gemm2Params_N_O_M::BThreadClusterLengths,
            typename Gemm2Params_N_O_M::BThreadClusterArrangeOrder,
            InputDataType,
            GemmDataType,
            GridDesc_M0_O_M1,
            decltype(b_block_desc_m0_o_m1),
            typename Gemm2Params_N_O_M::BThreadClusterArrangeOrder, // access order == thread order
            Sequence<1, 0, 2>,
            Gemm2Params_N_O_M::BSrcVectorDim,
            2, // DstVectorDim
            Gemm2Params_N_O_M::BSrcScalarPerVector,
            Gemm2Params_N_O_M::B_M1,
            1,
            1,
            true,
            true,
            1>;

        using BlockwiseGemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                GemmDataType,
                                                                FloatGemmAcc,
                                                                decltype(a_block_desc_m0_n_m1),
                                                                decltype(b_block_desc_m0_o_m1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                Gemm2Params_N_O_M::GemmNRepeat,
                                                                Gemm2Params_N_O_M::GemmORepeat,
                                                                Gemm2Params_N_O_M::GemmMPack,
                                                                true>; // TranspossC

        static constexpr auto b_block_slice_copy_step =
            make_multi_index(Gemm2Params_N_O_M::B_M0, 0, 0);
        static constexpr auto c_block_slice_copy_step =
            make_multi_index(Gemm2Params_N_O_M::GemmNRepeat, 0, 0, 0, 0, 0, 0, 0);
        static constexpr auto b_block_reset_copy_step =
            make_multi_index(-MPerBlock / Gemm2Params_N_O_M::B_M1, 0, 0);

        template <typename CGradDesc_N_O>
        __host__ __device__ static auto
        MakeCGridDesc_N0_O0_N1_O1_N2_O2_O3_O4(const CGradDesc_N_O& c_grid_desc_n_o)
        {
            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            const auto c_grid_desc_n0_o0_n1_o1_n2_o2 = transform_tensor_descriptor(
                c_grid_desc_n_o,
                make_tuple(
                    make_unmerge_transform(make_tuple(I1, Gemm2Params_N_O_M::GemmNWave, MPerXdl)),
                    make_unmerge_transform(make_tuple(I1, Gemm2Params_N_O_M::GemmOWave, NPerXdl))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

            const auto c_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
                BlockwiseGemm{}.xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(
                    c_grid_desc_n0_o0_n1_o1_n2_o2);

            return c_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4;
        }

        static constexpr auto c_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            BlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        __host__ __device__ static auto GetCThreadOriginOnBlock_N0_O0_N1_O1_N2_O2_O3_O4()
        {
            return to_multi_index(BlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0));
        }

        template <typename CGridDesc_N0_O0_N1_O1_N2_O2_O3_O4,
                  typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using CBlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            OutputDataType,
            decltype(c_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4),
            CGridDesc_N0_O0_N1_O1_N2_O2_O3_O4,
            ElementwiseOp,                                                // CElementwiseOperation
            decltype(c_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLengths()), // SliceLengths
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,                             // AccessOrder
            7,                                                            // VectorDim
            2,                                                            // ScalarPerVector
            InMemoryDataOperationEnum::AtomicAdd, // GlobalMemoryDataOperation
            1,                                    // DstScalarStrideInVector
            true>;
    };

    template <index_t BlockSize_, index_t BlockSliceLength_M_, index_t BlockSliceLength_O_>
    struct YDotYGrad_M_O_
    {
        static constexpr index_t SrcScalarPerVector = 16 / sizeof(InputDataType);
        static constexpr auto ThreadClusterLength_O =
            Number<BlockSliceLength_O_ / SrcScalarPerVector>{};
        static constexpr auto ThreadClusterLength_M = Number<BlockSize_ / ThreadClusterLength_O>{};
        static constexpr auto ThreadSliceLength_O   = Number<SrcScalarPerVector>{};
        static constexpr auto ThreadSliceLength_M =
            Number<BlockSliceLength_M_ * ThreadClusterLength_O / BlockSize_>{};

        static_assert(ThreadClusterLength_O * ThreadSliceLength_O == BlockSliceLength_O_, "");
        static_assert(ThreadClusterLength_M * ThreadSliceLength_M == BlockSliceLength_M_, "");

        using SrcBufType = StaticBuffer<AddressSpaceEnum::Vgpr,
                                        FloatGemmAcc,
                                        ThreadSliceLength_M * ThreadSliceLength_O,
                                        true>;

        using DstBufType =
            StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, ThreadSliceLength_M, true>;
    };
    using YDotYGrad_M_O = YDotYGrad_M_O_<BlockSize, MPerBlock, Gemm1NPerBlock>;

    // PGrad Gemm has the same layout as P = Q * K^T Gemm (A row-major B col-major)
    struct PGradGemmTile_M_N_O
    {
        // TODO:
        // Make all input tensors 2D and transform them into appropriate 3D form in kernel to make
        // things more concise
        template <typename YGradGridDesc_M0_O_M1_>
        __device__ static auto
        MakeYGradGridDesc_O0_M_O1(const YGradGridDesc_M0_O_M1_& ygrad_grid_desc_m0_o_m1)
        {
            const auto M0 = ygrad_grid_desc_m0_o_m1.GetLength(I0);
            const auto O  = ygrad_grid_desc_m0_o_m1.GetLength(I1);
            const auto M1 = ygrad_grid_desc_m0_o_m1.GetLength(I2);

            constexpr auto Y_O1 = AK1;
            const auto Y_O0     = O / Y_O1;

            const auto ygrad_grid_desc_o0_m_o1 = transform_tensor_descriptor(
                ygrad_grid_desc_m0_o_m1,
                make_tuple(make_unmerge_transform(make_tuple(Y_O0, Y_O1)),
                           make_merge_transform_v3_division_mod(make_tuple(M0, M1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ygrad_grid_desc_o0_m_o1;
        }

        template <typename VGridDesc_N0_O_N1_>
        __device__ static auto MakeVGridDesc_O0_N_O1(const VGridDesc_N0_O_N1_& v_grid_desc_n0_o_n1)
        {
            const auto N0 = v_grid_desc_n0_o_n1.GetLength(I0);
            const auto O  = v_grid_desc_n0_o_n1.GetLength(I1);
            const auto N1 = v_grid_desc_n0_o_n1.GetLength(I2);

            constexpr auto V_O1 = BK1;
            const auto V_O0     = O / V_O1;

            const auto v_grid_desc_o0_n_o1 = transform_tensor_descriptor(
                v_grid_desc_n0_o_n1,
                make_tuple(make_unmerge_transform(make_tuple(V_O0, V_O1)),
                           make_merge_transform_v3_division_mod(make_tuple(N0, N1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return v_grid_desc_o0_n_o1;
        }
    };

    // QGrad Gemm has the same layout as Y = P * V Gemm (A in acc B row-major)
    struct QGradGemmTile_M_K_N
    {
        template <typename QGridDesc_K0_M_K1_>
        __device__ static auto MakeQGradGridDesc_MBlock_MPerBlock_KBlock_KPerBlock(
            const QGridDesc_K0_M_K1_& q_grid_desc_k0_m_k1)
        {
            const auto K0 = q_grid_desc_k0_m_k1.GetLength(I0);
            const auto M  = q_grid_desc_k0_m_k1.GetLength(I1);
            const auto K1 = q_grid_desc_k0_m_k1.GetLength(I2);
            const auto K  = K0 * K1;

            const auto MBlock = M / MPerBlock;
            const auto KBlock = K / Gemm1NPerBlock; // NOTE: QGrad gemm is similar to Y gemm

            const auto q_grid_desc_m_k = transform_tensor_descriptor(
                q_grid_desc_k0_m_k1,
                make_tuple(make_pass_through_transform(M),
                           make_merge_transform_v3_division_mod(make_tuple(K0, K1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                q_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                           make_unmerge_transform(make_tuple(KBlock, Number<Gemm1NPerBlock>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
        }

        template <typename KGridDesc_K0_N_K1_>
        __device__ static auto MakeKGridDesc_N0_K_N1(const KGridDesc_K0_N_K1_& k_grid_desc_k0_n_k1)
        {
            const auto K_K0 = k_grid_desc_k0_n_k1.GetLength(I0);
            const auto N    = k_grid_desc_k0_n_k1.GetLength(I1);
            const auto K_K1 = k_grid_desc_k0_n_k1.GetLength(I2);

            constexpr auto K_N1 = B1K1;
            const auto K_N0     = N / K_N1;

            const auto k_grid_desc_n0_k_n1 = transform_tensor_descriptor(
                k_grid_desc_k0_n_k1,
                make_tuple(make_unmerge_transform(make_tuple(K_N0, K_N1)),
                           make_merge_transform_v3_division_mod(make_tuple(K_K0, K_K1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return k_grid_desc_n0_k_n1;
        }
    };

    struct KGradGemmTile_N_K_M
    {
        // B position
        template <typename QGridDesc_K0_M_K1_>
        __device__ static auto MakeQGridDesc_M0_K_M1(const QGridDesc_K0_M_K1_& q_grid_desc_k0_m_k1)
        {
            const auto Q_K0 = q_grid_desc_k0_m_k1.GetLength(I0);
            const auto M    = q_grid_desc_k0_m_k1.GetLength(I1);
            const auto Q_K1 = q_grid_desc_k0_m_k1.GetLength(I2);

            constexpr auto Q_M1 = B1K1;
            const auto Q_M0     = M / Q_M1;

            const auto q_grid_desc_m0_k_m1 = transform_tensor_descriptor(
                q_grid_desc_k0_m_k1,
                make_tuple(make_unmerge_transform(make_tuple(Q_M0, Q_M1)),
                           make_merge_transform_v3_division_mod(make_tuple(Q_K0, Q_K1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return q_grid_desc_m0_k_m1;
        }

        // C position
        template <typename KGridDesc_K0_N_K1_>
        __device__ static auto MakeKGradGridDesc_N_K(const KGridDesc_K0_N_K1_& k_grid_desc_k0_n_k1)
        {
            const auto K_K0 = k_grid_desc_k0_n_k1.GetLength(I0);
            const auto N    = k_grid_desc_k0_n_k1.GetLength(I1);
            const auto K_K1 = k_grid_desc_k0_n_k1.GetLength(I2);

            return transform_tensor_descriptor(
                k_grid_desc_k0_n_k1,
                make_tuple(make_pass_through_transform(N),
                           make_merge_transform_v3_division_mod(make_tuple(K_K0, K_K1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    };

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto a2_block_desc_m0_n_m1 =
            GetA2BlockDescriptor_M0_N_M1<Gemm2Params_N_O_M>();
        static constexpr auto b2_block_desc_m0_o_m1 =
            GetB2BlockDescriptor_M0_O_M1<Gemm2Params_N_O_M>();

        static constexpr auto max_lds_align = Number<16 / sizeof(GemmDataType)>{};

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto p_block_space_size_aligned = math::integer_least_multiple(
            a2_block_desc_m0_n_m1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto ygrad_block_space_size_aligned = math::integer_least_multiple(
            b2_block_desc_m0_o_m1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset  = 0;
        static constexpr auto b_block_space_offset  = a_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = 0;
        static constexpr auto a2_block_space_offset = 0;
        static constexpr auto b2_block_space_offset = p_block_space_size_aligned.value;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a_block_space_size_aligned +
                                         SharedMemTrait::b_block_space_size_aligned) *
                                        sizeof(GemmDataType);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(GemmDataType);
        const index_t vgrad_gemm_bytes_end = (SharedMemTrait::p_block_space_size_aligned +
                                              SharedMemTrait::ygrad_block_space_size_aligned) *
                                             sizeof(GemmDataType);

        const index_t softmax_bytes_end = (SharedMemTrait::reduction_space_offset +
                                           SharedMemTrait::reduction_space_size_aligned) *
                                          sizeof(FloatGemmAcc);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(FloatCShuffle);

        return math::max(gemm0_bytes_end,
                         gemm1_bytes_end,
                         vgrad_gemm_bytes_end,
                         softmax_bytes_end,
                         c_block_bytes_end);
    }

    template <bool HasMainKBlockLoop,
              typename Block2CTileMap,
              typename C0MatrixMask,
              typename VGradGridDescriptor_N_O,
              typename YGradGridDesc_M0_O_M1>
    __device__ static void Run(const InputDataType* __restrict__ p_q_grid,
                               const InputDataType* __restrict__ p_k_grid,
                               ZDataType* __restrict__ p_z_grid,
                               const InputDataType* __restrict__ p_v_grid,
                               const InputDataType* __restrict__ p_y_grid,
                               const FloatLSE* __restrict__ p_lse_grid,
                               const InputDataType* __restrict__ p_ygrad_grid,
                               OutputDataType* __restrict__ p_qgrad_grid,
                               OutputDataType* __restrict__ p_kgrad_grid,
                               OutputDataType* __restrict__ p_vgrad_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const SElementwiseOperation& s_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
                               const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
                               const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5&
                                   z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                               const VGridDesc_N0_O_N1& v_grid_desc_n0_o_n1,
                               const YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock&
                                   y_grid_desc_mblock_mperblock_oblock_operblock,
                               const LSEGridDesc_M& lse_grid_desc_m,
                               const VGradGridDescriptor_N_O& vgrad_grid_desc_n_o,
                               const YGradGridDesc_M0_O_M1& ygrad_grid_desc_m0_o_m1,
                               const Block2CTileMap& block_2_ctile_map,
                               const C0MatrixMask& c0_matrix_mask,
                               const float p_drop,
                               ck::philox& ph,
                               const index_t block_idx_m)
    {
        const FloatGemmAcc p_dropout  = type_convert<FloatGemmAcc>(1.0f - p_drop);
        const FloatGemmAcc rp_dropout = type_convert<FloatGemmAcc>(1.0f / p_dropout);
        const ushort p_dropout_in_16bits =
            __builtin_amdgcn_readfirstlane(std::floor(p_dropout * 65535.0));
        const tensor_operation::element_wise::Scale scale_rp_dropout(s_element_op.Value() *
                                                                     rp_dropout);

        const auto q_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_q_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto k_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_k_grid, k_grid_desc_k0_n_k1.GetElementSpaceSize());
        const auto v_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_v_grid, v_grid_desc_n0_o_n1.GetElementSpaceSize());
        const auto y_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_grid, y_grid_desc_mblock_mperblock_oblock_operblock.GetElementSpaceSize());
        const auto lse_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_lse_grid, lse_grid_desc_m.GetElementSpaceSize());
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, ygrad_grid_desc_m0_o_m1.GetElementSpaceSize());
        auto vgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_vgrad_grid, vgrad_grid_desc_n_o.GetElementSpaceSize());
        auto qgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_qgrad_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        auto kgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_kgrad_grid, k_grid_desc_k0_n_k1.GetElementSpaceSize());

        // divide block work by [M, O]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I0),
                          y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_work_idx_m = Deterministic ? block_idx_m : block_work_idx[I0];

        // HACK: this force m/o_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx_m * MPerBlock);

        const index_t o_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1NPerBlock);

        // 6 GEMM operations are categorized into 3 buckets. SizeK == SizeO == head_dim
        // S_MNK / dP_MNO Gemm (Gemm0 rcr)
        // Y_MON / dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)

        //
        // set up S / dP Gemm (type 1 rcr)
        //

        // Gemm0: LDS allocation for A and B: be careful of alignment
        auto gemm0_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::a_block_space_offset,
            Gemm0::a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto gemm0_b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::b_block_space_offset,
            Gemm0::b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // Gemm0: gridwise GEMM pipeline
        // Only supports LoopScheduler::Default
        const auto gemm0_gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer,
                                          NumGemmKPrefetchStage,
                                          LoopScheduler::Default>();
        // S: A matrix blockwise copy
        auto s_gemm_tile_q_blockwise_copy =
            typename Gemm0::template ABlockwiseCopy<decltype(q_grid_desc_k0_m_k1)>(
                q_grid_desc_k0_m_k1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                Gemm0::a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // S: B matrix blockwise copy
        auto s_gemm_tile_k_blockwise_copy =
            typename Gemm0::template BBlockwiseCopy<decltype(k_grid_desc_k0_n_k1)>(
                k_grid_desc_k0_n_k1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                b_element_op,
                Gemm0::b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // S: blockwise gemm
        auto s_blockwise_gemm = typename Gemm0::BlockwiseGemm{}; // TransposeC

        auto s_slash_p_thread_buf = s_blockwise_gemm.GetCThreadBuffer();

        const auto s_gemm_tile_a_block_reset_copy_step =
            make_multi_index(-q_grid_desc_k0_m_k1.GetLength(I0), 0, 0);
        const auto s_gemm_tile_b_block_reset_copy_step =
            make_multi_index(-k_grid_desc_k0_n_k1.GetLength(I0), NPerBlock, 0);

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (q_grid_desc_k0_m_k1.GetLength(I0) * q_grid_desc_k0_m_k1.GetLength(I2)) / KPerBlock);

        // dP: transform input and output tensor descriptors
        const auto ygrad_grid_desc_o0_m_o1 =
            PGradGemmTile_M_N_O::MakeYGradGridDesc_O0_M_O1(ygrad_grid_desc_m0_o_m1);
        const auto v_grid_desc_o0_n_o1 =
            PGradGemmTile_M_N_O::MakeVGridDesc_O0_N_O1(v_grid_desc_n0_o_n1);

        // dP: A matrix blockwise copy
        auto pgrad_gemm_tile_ygrad_blockwise_copy =
            typename Gemm0::template ABlockwiseCopy<decltype(ygrad_grid_desc_o0_m_o1)>(
                ygrad_grid_desc_o0_m_o1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                tensor_operation::element_wise::PassThrough{},
                Gemm0::a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dP: B matrix blockwise copy
        auto pgrad_gemm_tile_v_blockwise_copy =
            typename Gemm0::template BBlockwiseCopy<decltype(v_grid_desc_o0_n_o1)>(
                v_grid_desc_o0_n_o1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                tensor_operation::element_wise::PassThrough{},
                Gemm0::b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dP: blockwise gemm
        // we need separate blockwise gemm object because we need separate thread buffer
        auto pgrad_blockwise_gemm = typename Gemm0::BlockwiseGemm{};

        auto pgrad_thread_buf = pgrad_blockwise_gemm.GetCThreadBuffer();

        const auto pgrad_gemm_tile_ygrad_block_reset_copy_step =
            make_multi_index(-ygrad_grid_desc_o0_m_o1.GetLength(I0), 0, 0);
        const auto pgrad_gemm_tile_v_block_reset_copy_step =
            make_multi_index(-v_grid_desc_o0_n_o1.GetLength(I0), NPerBlock, 0);

        const index_t num_o_block_main_loop = __builtin_amdgcn_readfirstlane(
            (ygrad_grid_desc_o0_m_o1.GetLength(I0) * ygrad_grid_desc_o0_m_o1.GetLength(I2)) /
            KPerBlock);

        //
        // set up Y / dQ Gemm (type 2 rrr)
        //
        // Note: Y is pre-calculated in forward pass and loaded to backward pass kernel
        using Gemm1 =
            Gemm1<decltype(s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()),
                  decltype(s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4())>;

        // Gemm1: VGPR allocation for A and LDS allocation for B
        auto gemm1_a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
            Gemm1::a_thread_desc_k0_m_k1.GetElementSpaceSize());

        auto gemm1_b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            Gemm1::b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // dQ: transform input and output tensor descriptors
        const auto k_grid_desc_n0_k_n1 =
            QGradGemmTile_M_K_N::MakeKGridDesc_N0_K_N1(k_grid_desc_k0_n_k1);
        auto qgrad_grid_desc_mblock_mperblock_kblock_kperblock =
            QGradGemmTile_M_K_N::MakeQGradGridDesc_MBlock_MPerBlock_KBlock_KPerBlock(
                q_grid_desc_k0_m_k1);

        // dQ: A matrix blockwise copy
        auto qgrad_gemm_tile_sgrad_blockwise_copy =
            typename Gemm1::ABlockwiseCopy{tensor_operation::element_wise::PassThrough{}};

        // dQ: B matrix blockwise copy
        auto qgrad_gemm_tile_k_blockwise_copy =
            typename Gemm1::template BBlockwiseCopy<decltype(k_grid_desc_n0_k_n1)>(
                k_grid_desc_n0_k_n1,
                make_multi_index(0, o_block_data_idx_on_grid, 0),
                b1_element_op,
                Gemm1::b_block_desc_bk0_n_bk1, // there n actually is k, k is N, so name can be
                                               // b_block_desc_bn0_k_bn1
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dQ: blockwise gemm
        auto qgrad_blockwise_gemm =
            typename Gemm1::BlockwiseGemm{make_tuple(0, 0, 0, 0)}; // A_origin

        auto qgrad_thread_buf = qgrad_blockwise_gemm.GetCThreadBuffer();

        //
        // Blockwise softmax
        //
        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4 =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths() /
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I4);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I5);
        constexpr auto tn3 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I6);
        constexpr auto tn4 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I7);

        // get acc0 thread map
        constexpr auto m0_n_m1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tm0 * tm1, tm2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_m0_n_m1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tm0 * tm1, tn0 * tn1 * tn2 * tn3 * tn4, tm2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(m0_n_m1_to_m_n_adaptor, threadid_to_m0_n_m1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
        constexpr auto m0 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto n0 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto m1 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto n1 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto m2 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto n2 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto n3 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto n4 = thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2, tn0 * tn1 * tn2 * tn3 * tn4));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2, n0 * n1 * n2 * n3 * n4));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        auto blockwise_dropout = BlockwiseDropout<FloatGemmAcc, decltype(thread_slice_desc_m_n)>{
            p_dropout_in_16bits, rp_dropout};

        auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl =
            MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(lse_grid_desc_m);

        constexpr auto lse_thread_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor_packed(make_tuple(I1, m0, m1, m2));

        auto lse_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatLSE>(
            lse_thread_desc_mblock_mrepeat_mwave_mperxdl.GetElementSpaceSize());

        auto acc0_thread_origin = s_blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

        auto lse_thread_copy_global_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatLSE,
                                             FloatLSE,
                                             decltype(lse_grid_desc_mblock_mrepeat_mwave_mperxdl),
                                             decltype(lse_thread_desc_mblock_mrepeat_mwave_mperxdl),
                                             Sequence<1, m0, m1, m2>,
                                             Sequence<0, 1, 2, 3>,
                                             3,
                                             m2,
                                             1,
                                             false>{
                lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                make_multi_index(block_work_idx_m,       // mblock
                                 acc0_thread_origin[I0],   // mrepeat
                                 acc0_thread_origin[I2],   // mwave
                                 acc0_thread_origin[I4])}; // mperxdl

        //
        // z vgpr copy to global
        //
        // z matrix threadwise desc
        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockID
                                                           m0,   // MRepeat
                                                           I1,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MPerXdl
                                                           n2,   // NGroupNum
                                                           n3,   // NInputNum
                                                           n4)); // registerNum

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     unsigned short,
                     z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize(),
                     true>
            z_tenor_buffer;
        z_tenor_buffer.Clear();
        // z matrix global desc
        /*const auto M = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto N = k_grid_desc_k0_n_k1.GetLength(I1);

        auto z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            MakeZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(M, N);*/

        auto z_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_z_grid, z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize());

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        auto z_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            ushort,
            ZDataType,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            decltype(z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, // MBlockId
                     I1, // NBlockID
                     m0, // MRepeat
                     I1, // NRepeat
                     m1, // MWaveId
                     n1, // NWaveId
                     m2, // MPerXdl
                     n2, // NGroupNum
                     n3, // NInputNum
                     n4>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
            9, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                  make_multi_index(block_work_idx_m, // MBlockId
                                   0,                  // NBlockId
                                   0,                  // mrepeat
                                   0,                  // nrepeat
                                   wave_id[I0],        // MWaveId
                                   wave_id[I1],        // NWaveId
                                   wave_m_n_id[I1],    // MPerXdl
                                   0,                  // group
                                   wave_m_n_id[I0],    // NInputIndex
                                   0),
                  tensor_operation::element_wise::PassThrough{}};

        //
        // set up dV / dK Gemm (type 3 crr)
        //
        using Gemm2 = Gemm2<Gemm2Params_N_O_M, decltype(s_blockwise_gemm)>;

        // Gemm2: LDS allocation for A and B: be careful of alignment
        auto gemm2_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::a2_block_space_offset,
            Gemm2::a_block_desc_m0_n_m1.GetElementSpaceSize());

        auto gemm2_b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::b2_block_space_offset,
            Gemm2::b_block_desc_m0_o_m1.GetElementSpaceSize());

        // dV: transform input and output tensor descriptors
        const auto vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            Gemm2::MakeCGridDesc_N0_O0_N1_O1_N2_O2_O3_O4(vgrad_grid_desc_n_o);

        // dV: A matrix VGPR-to-LDS blockwise copy
        auto vgrad_gemm_tile_p_thread_copy_vgpr_to_lds =
            typename Gemm2::template ABlockwiseCopy<tensor_operation::element_wise::Relu>{
                Gemm2::a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                Gemm2::MakeAThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4(),
                tensor_operation::element_wise::Relu{}}; // relu(P-dropped)

        // dV: B matrix global-to-LDS blockwise copy
        auto vgrad_gemm_tile_ygrad_blockwise_copy =
            typename Gemm2::template BBlockwiseCopy<decltype(ygrad_grid_desc_m0_o_m1)>(
                ygrad_grid_desc_m0_o_m1,
                make_multi_index(m_block_data_idx_on_grid / Gemm2Params_N_O_M::B_M1,
                                 o_block_data_idx_on_grid,
                                 0),
                tensor_operation::element_wise::PassThrough{},
                Gemm2::b_block_desc_m0_o_m1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dV: blockwise gemm
        auto v_slash_k_grad_blockwise_gemm = typename Gemm2::BlockwiseGemm{};

        auto v_slash_k_grad_thread_buf = v_slash_k_grad_blockwise_gemm.GetCThreadBuffer();

        // dV: C VGPR-to-global copy
        const auto vgrad_thread_origin_on_grid_n0_o0_n1_o1_n2_o2_o3_o4 =
            Gemm2::GetCThreadOriginOnBlock_N0_O0_N1_O1_N2_O2_O3_O4() +
            make_multi_index(
                I0, block_work_idx[I1] * Gemm2Params_N_O_M::GemmORepeat, I0, I0, I0, I0, I0, I0);

        auto vgrad_thread_copy_vgpr_to_global =
            typename Gemm2::template CBlockwiseCopy<decltype(
                                                        vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4),
                                                    tensor_operation::element_wise::Scale>(
                vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                vgrad_thread_origin_on_grid_n0_o0_n1_o1_n2_o2_o3_o4,
                tensor_operation::element_wise::Scale{rp_dropout});

        // dK: transform input and output tensor descriptors
        const auto q_grid_desc_m0_k_m1 =
            KGradGemmTile_N_K_M::MakeQGridDesc_M0_K_M1(q_grid_desc_k0_m_k1);
        const auto kgrad_grid_desc_n_k =
            KGradGemmTile_N_K_M::MakeKGradGridDesc_N_K(k_grid_desc_k0_n_k1);
        const auto kgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            Gemm2::MakeCGridDesc_N0_O0_N1_O1_N2_O2_O3_O4(kgrad_grid_desc_n_k);

        // dK: A matrix VGPR-to-LDS blockwise copy
        auto kgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds =
            typename Gemm2::template ABlockwiseCopy<tensor_operation::element_wise::PassThrough>{
                Gemm2::a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                Gemm2::MakeAThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4(),
                tensor_operation::element_wise::PassThrough{}};

        // dK: B matrix global-to-LDS blockwise copy
        auto kgrad_gemm_tile_q_blockwise_copy =
            typename Gemm2::template BBlockwiseCopy<decltype(q_grid_desc_m0_k_m1)>(
                q_grid_desc_m0_k_m1,
                make_multi_index(m_block_data_idx_on_grid / Gemm2Params_N_O_M::B_M1,
                                 o_block_data_idx_on_grid,
                                 0),
                tensor_operation::element_wise::PassThrough{},
                Gemm2::b_block_desc_m0_o_m1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dK: blockwise gemm
        /* reuse v_slash_k_grad_blockwise_gemm, v_slash_k_grad_thread_buf */

        // dK: C VGPR-to-global copy
        const auto kgrad_thread_origin_on_grid_n0_o0_n1_o1_n2_o2_o3_o4 =
            Gemm2::GetCThreadOriginOnBlock_N0_O0_N1_O1_N2_O2_O3_O4() +
            make_multi_index(
                I0, block_work_idx[I1] * Gemm2Params_N_O_M::GemmORepeat, I0, I0, I0, I0, I0, I0);

        auto kgrad_thread_copy_vgpr_to_global = typename Gemm2::template CBlockwiseCopy<
            decltype(kgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4),
            decltype(scale_rp_dropout)>(kgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                        kgrad_thread_origin_on_grid_n0_o0_n1_o1_n2_o2_o3_o4,
                                        scale_rp_dropout);

        //
        // set up Y dot dY
        //

        // m0, n0 are m/n repeat per wave
        // m1, n1 are number of waves
        constexpr auto p_block_lengths =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto P_M0 = p_block_lengths[I0]; // repeats
        constexpr auto P_M1 = p_block_lengths[I2]; // waves
        constexpr auto P_M2 = p_block_lengths[I4]; // xdl

        constexpr auto y_thread_desc_m0_m1_o0_o1 = make_naive_tensor_descriptor_packed(make_tuple(
            I1, YDotYGrad_M_O::ThreadSliceLength_M, I1, YDotYGrad_M_O::ThreadSliceLength_O));
        constexpr auto y_thread_cluster_desc =
            make_cluster_descriptor(Sequence<I1,
                                             YDotYGrad_M_O::ThreadClusterLength_M,
                                             I1,
                                             YDotYGrad_M_O::ThreadClusterLength_O>{},
                                    Sequence<0, 1, 2, 3>{});
        const auto y_thread_cluster_idx =
            y_thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto y_thread_data_on_block_idx =
            y_thread_cluster_idx * y_thread_desc_m0_m1_o0_o1.GetLengths();
        const auto y_thread_data_on_grid_idx =
            make_multi_index(
                block_work_idx_m, I0, I0 /* all WGs start from o_block_idx = 0 */, I0) +
            y_thread_data_on_block_idx;

        // performs double duty for both y and ygrad
        auto yygrad_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            InputDataType,
            FloatGemmAcc,
            YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock,
            decltype(y_thread_desc_m0_m1_o0_o1),
            decltype(y_thread_desc_m0_m1_o0_o1.GetLengths()),
            Sequence<0, 1, 2, 3>,
            3,                                 // SrcVectorDim
            YDotYGrad_M_O::SrcScalarPerVector, // SrcScalarPerVector
            1,                                 // SrcScalarStrideInVector
            true /* ResetCoordAfterRun */,
            false /* InvalidElementAsNaN */>(y_grid_desc_mblock_mperblock_oblock_operblock,
                                             y_thread_data_on_grid_idx);

        auto y_thread_buf                 = typename YDotYGrad_M_O::SrcBufType{};
        auto ygrad_thread_buf             = typename YDotYGrad_M_O::SrcBufType{};
        auto y_dot_ygrad_thread_accum_buf = typename YDotYGrad_M_O::DstBufType{};
        auto y_dot_ygrad_block_accum_buf  = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared), MPerBlock);

        constexpr auto y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor(make_tuple(I1, P_M0, P_M1, P_M2),
                                         make_tuple(P_M0 * P_M1 * P_M2, P_M1 * P_M2, P_M2, I1));
        constexpr auto y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl =
            lse_thread_desc_mblock_mrepeat_mwave_mperxdl; // reuse LSE thread descriptor because
                                                          // per-thread LSE data and y_dot_ygrad is
                                                          // tiled the same way

        auto y_dot_ygrad_thread_copy_lds_to_vgpr = ThreadwiseTensorSliceTransfer_v2<
            FloatGemmAcc,
            FloatGemmAcc,
            decltype(y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl),
            decltype(y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl),
            Sequence<1, m0, m1, m2>,
            Sequence<0, 1, 2, 3>,
            3,
            m2,
            1,
            false>{y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl,
                   make_multi_index(I0,                       // mblock
                                    acc0_thread_origin[I0],   // mrepeat
                                    acc0_thread_origin[I2],   // mwave
                                    acc0_thread_origin[I4])}; // mperxdl

        auto y_dot_ygrad_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatGemmAcc>(
            y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl.GetElementSpaceSize());

        if constexpr(Deterministic)
        {
            block_sync_lds();
        }

        //
        // calculate Y dot dY
        //

        // clear accum buffers
        y_dot_ygrad_thread_accum_buf.Clear();
        y_dot_ygrad_block_accum_buf.Clear();

        index_t oblock_idx = 0;
        do
        {
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       y_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       y_thread_buf);
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       ygrad_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       ygrad_thread_buf);

            static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
                static_for<0, YDotYGrad_M_O::ThreadSliceLength_O, 1>{}([&](auto iO) {
                    constexpr auto offset =
                        y_thread_desc_m0_m1_o0_o1.CalculateOffset(make_multi_index(I0, iM, I0, iO));
                    y_dot_ygrad_thread_accum_buf(iM) +=
                        y_thread_buf[Number<offset>{}] * ygrad_thread_buf[Number<offset>{}];
                });
            });

            yygrad_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_oblock_operblock,
                                                      make_multi_index(0, 0, 1, 0));

            oblock_idx++;
        } while(oblock_idx < y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2));

        // blockwise reduction using atomic_add
        block_sync_lds();
        static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
            const auto idx_on_block = y_thread_data_on_block_idx[I1] + iM;
            y_dot_ygrad_block_accum_buf.AtomicAdd(
                idx_on_block, true, y_dot_ygrad_thread_accum_buf[iM] * p_dropout); // p_dropoutD1
        });
        block_sync_lds();

        // distribute y_dot_ygrad to threads; LDS accum buffer can be safely reused after barrier
        y_dot_ygrad_thread_copy_lds_to_vgpr.Run(
            y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl,
            y_dot_ygrad_block_accum_buf,
            y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl,
            make_tuple(I0, I0, I0, I0),
            y_dot_ygrad_thread_buf);

        lse_thread_copy_global_to_vgpr.Run(lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                                           lse_grid_buf,
                                           lse_thread_desc_mblock_mrepeat_mwave_mperxdl,
                                           make_tuple(I0, I0, I0, I0),
                                           lse_thread_buf);

        const index_t num_gemm1_k_block_outer_loop = k_grid_desc_k0_n_k1.GetLength(I1) / NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = NPerBlock / Gemm1KPerBlock;

        // Initialize dQ
        qgrad_thread_buf.Clear();

        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
        do
        {
            auto n_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm1_k_block_outer_index * NPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, MPerBlock, NPerBlock))
            {
                continue;
            }
            // S = Q * K^T
            gemm0_gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(
                q_grid_desc_k0_m_k1,
                Gemm0::a_block_desc_ak0_m_ak1,
                s_gemm_tile_q_blockwise_copy,
                q_grid_buf,
                gemm0_a_block_buf,
                Gemm0::a_block_slice_copy_step,
                k_grid_desc_k0_n_k1,
                Gemm0::b_block_desc_bk0_n_bk1,
                s_gemm_tile_k_blockwise_copy,
                k_grid_buf,
                gemm0_b_block_buf,
                Gemm0::b_block_slice_copy_step,
                s_blockwise_gemm,
                s_slash_p_thread_buf,
                num_k_block_main_loop);

            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {
                // 8d thread_desc in thread scope
                constexpr auto c_thread_lengths =
                    s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                // 8d block_desc in block scope
                constexpr auto c_block_lengths =
                    s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                constexpr auto M0 = c_block_lengths[I0];
                constexpr auto N0 = c_block_lengths[I1];
                constexpr auto M1 = c_block_lengths[I2];
                constexpr auto N1 = c_block_lengths[I3];
                constexpr auto M2 = c_block_lengths[I4];
                constexpr auto N2 = c_block_lengths[I5];
                constexpr auto N3 = c_block_lengths[I6];
                constexpr auto N4 = c_block_lengths[I7];

                // works like multi-dimension static_for (static_ford), but provides both the linear
                // index as well as n-d index
                using Acc0TileIterator = SpaceFillingCurve<
                    decltype(c_thread_lengths),
                    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                    false>; // SnakeCurved

                constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2)),
                               make_unmerge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));

                static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;
                    if(c0_matrix_mask.IsMaskedElement(m_global, n_global))
                    {
                        s_slash_p_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
                    }
                    else
                    {
                        s_element_op(s_slash_p_thread_buf(i), s_slash_p_thread_buf[i]);
                    }
                });
            }
            else
            {
                static_for<0, s_slash_p_thread_buf.Size(), 1>{}([&](auto i) {
                    s_element_op(s_slash_p_thread_buf(i), s_slash_p_thread_buf[i]);
                });
            }

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

            // P_i: = softmax(scalar * S_i:)
            // scaling is already performed in the preceding statements with s_element_op
            blockwise_softmax.RunWithPreCalcStats(s_slash_p_thread_buf, lse_thread_buf);

            // save z to global
            if(p_z_grid)
            {
                // P_dropped
                static_for<0, n0, 1>{}([&](auto i) {
                    blockwise_dropout.template ApplyDropout<decltype(s_slash_p_thread_buf),
                                                            decltype(z_tenor_buffer),
                                                            true,
                                                            decltype(n0),
                                                            decltype(i)>(
                        s_slash_p_thread_buf, ph, z_tenor_buffer);

                    z_thread_copy_vgpr_to_global.Run(
                        z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                        z_tenor_buffer,
                        z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        z_grid_buf);
                    z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                        z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        make_multi_index(0, 0, 0, 1, 0, 0, 0, 0, 0, 0));
                });
                z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                    make_multi_index(0, 0, 0, -n0.value, 0, 0, 0, 0, 0, 0));
            }
            else
            {
                ignore = z_grid_buf;
                // P_dropped
                blockwise_dropout.template ApplyDropout<decltype(s_slash_p_thread_buf), true>(
                    s_slash_p_thread_buf, ph);
            }

            block_sync_lds(); // wait for gemm1 LDS read

            SubThreadBlock<BlockSize> gemm2_a_copy_subgroup(s_blockwise_gemm.GetWaveIdx()[I0],
                                                            s_blockwise_gemm.GetWaveIdx()[I1]);

            constexpr index_t num_gemm2_loop = MPerBlock / Gemm2Params_N_O_M::Sum_M;
            static_assert(Gemm2::ASrcBlockSliceWindowIterator::GetNumOfAccess() == num_gemm2_loop,
                          "");

            // TODO: tune gemm2 pipeline
            // dV = P_drop^T * dY
            v_slash_k_grad_thread_buf.Clear();
            static_for<0, num_gemm2_loop, 1>{}([&](auto gemm2_loop_idx) { // gemm dV
                // load VGrad Gemm B
                vgrad_gemm_tile_ygrad_blockwise_copy.RunRead(ygrad_grid_desc_m0_o_m1,
                                                             ygrad_grid_buf);

                // load VGrad Gemm A
                const auto p_slice_idx =
                    Gemm2::ASrcBlockSliceWindowIterator::GetIndexTupleOfNumber(gemm2_loop_idx);
                constexpr auto mwave_range = make_tuple(
                    p_slice_idx[I2],
                    p_slice_idx[I2] + Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(I2));
                constexpr auto nwave_range = make_tuple(
                    p_slice_idx[I3],
                    p_slice_idx[I3] + Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(I3));

                if(gemm2_a_copy_subgroup.IsBelong(mwave_range, nwave_range))
                {
                    vgrad_gemm_tile_p_thread_copy_vgpr_to_lds.Run(
                        Gemm2::a_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        make_tuple(p_slice_idx[I0], p_slice_idx[I1], I0, I0, I0, I0, I0, I0),
                        s_slash_p_thread_buf,
                        Gemm2::a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        gemm2_a_block_buf);
                }

                // ygrad slice window is moved with MoveSrcSliceWindow() since it is dynamic buffer
                // p slice window is moved by loop index
                vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_m0_o_m1, Gemm2::b_block_slice_copy_step);

                block_sync_lds(); // sync before write
                vgrad_gemm_tile_ygrad_blockwise_copy.RunWrite(Gemm2::b_block_desc_m0_o_m1,
                                                              gemm2_b_block_buf);

                block_sync_lds(); // sync before read
                v_slash_k_grad_blockwise_gemm.Run(
                    gemm2_a_block_buf, gemm2_b_block_buf, v_slash_k_grad_thread_buf);

            }); // end gemm dV
            // atomic_add dV
            vgrad_thread_copy_vgpr_to_global.Run(Gemm2::c_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                 v_slash_k_grad_thread_buf,
                                                 vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 vgrad_grid_buf);

            // gemm dP
            block_sync_lds();
            // dP = dY * V^T
            // assume size K == size O so HasMainKBlockLoop is the same
            gemm0_gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(
                ygrad_grid_desc_o0_m_o1,
                Gemm0::a_block_desc_ak0_m_ak1, // reuse
                pgrad_gemm_tile_ygrad_blockwise_copy,
                ygrad_grid_buf,
                gemm0_a_block_buf,              // reuse
                Gemm0::a_block_slice_copy_step, // reuse
                v_grid_desc_o0_n_o1,
                Gemm0::b_block_desc_bk0_n_bk1, // reuse
                pgrad_gemm_tile_v_blockwise_copy,
                v_grid_buf,
                gemm0_b_block_buf,              // reuse
                Gemm0::b_block_slice_copy_step, // reuse
                pgrad_blockwise_gemm,
                pgrad_thread_buf,
                num_o_block_main_loop);

            // dS = P * (dP - Y_dot_dY)
            auto& sgrad_thread_buf = pgrad_thread_buf;
            constexpr auto pgrad_thread_tile_iterator =
                pgrad_blockwise_gemm.MakeCThreadTileIterator();
            constexpr auto pgrad_thread_idx_to_m_n_adaptor =
                pgrad_blockwise_gemm.MakeCThreadIndexAdaptor8DTo2D();
            static_for<0, pgrad_thread_tile_iterator.GetNumOfAccess(), 1>{}([&](auto i) {
                constexpr auto pgrad_thread_idx = pgrad_thread_tile_iterator.GetIndex(i);
                constexpr auto m =
                    pgrad_thread_idx_to_m_n_adaptor.CalculateBottomIndex(pgrad_thread_idx)[I0];
                // dS and P has same thread buf layout
                if(s_slash_p_thread_buf[i] >= 0)
                {
                    sgrad_thread_buf(i) =
                        s_slash_p_thread_buf[i] *
                        (pgrad_thread_buf[i] - y_dot_ygrad_thread_buf[Number<m>{}]);
                }
                else
                {
                    sgrad_thread_buf(i) =
                        s_slash_p_thread_buf[i] * y_dot_ygrad_thread_buf[Number<m>{}];
                }
            });

            // gemm dQ
            // dQ = scalar * dS * K
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // preload data into LDS
                qgrad_gemm_tile_k_blockwise_copy.RunRead(k_grid_desc_n0_k_n1, k_grid_buf);

                qgrad_gemm_tile_k_blockwise_copy.MoveSrcSliceWindow(k_grid_desc_n0_k_n1,
                                                                    Gemm1::b_block_slice_copy_step);

                block_sync_lds(); // wait for previous LDS read

                qgrad_gemm_tile_k_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                          gemm1_b_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        qgrad_gemm_tile_sgrad_blockwise_copy.Run(Gemm1::a_src_thread_desc_k0_m_k1,
                                                                 Gemm1::a_block_slice_copy_step * i,
                                                                 sgrad_thread_buf,
                                                                 Gemm1::a_thread_desc_k0_m_k1,
                                                                 make_tuple(I0, I0, I0),
                                                                 gemm1_a_thread_buf);
                        qgrad_gemm_tile_k_blockwise_copy.RunRead(k_grid_desc_n0_k_n1, k_grid_buf);

                        block_sync_lds();

                        qgrad_blockwise_gemm.Run(
                            gemm1_a_thread_buf, gemm1_b_block_buf, qgrad_thread_buf);

                        block_sync_lds();

                        qgrad_gemm_tile_k_blockwise_copy.MoveSrcSliceWindow(
                            k_grid_desc_n0_k_n1, Gemm1::b_block_slice_copy_step);

                        qgrad_gemm_tile_k_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                                  gemm1_b_block_buf);
                    });
                }
                // tail
                {
                    qgrad_gemm_tile_sgrad_blockwise_copy.Run(
                        Gemm1::a_src_thread_desc_k0_m_k1,
                        Gemm1::a_block_slice_copy_step * Number<num_gemm1_k_block_inner_loop - 1>{},
                        sgrad_thread_buf,
                        Gemm1::a_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        gemm1_a_thread_buf);

                    block_sync_lds();

                    qgrad_blockwise_gemm.Run(
                        gemm1_a_thread_buf, gemm1_b_block_buf, qgrad_thread_buf);
                }
            } // end gemm dQ

            // dK = scalar * dS^T * Q
            v_slash_k_grad_thread_buf.Clear();
            static_for<0, num_gemm2_loop, 1>{}([&](auto gemm2_loop_idx) { // gemm dK
                // load KGrad Gemm B
                kgrad_gemm_tile_q_blockwise_copy.RunRead(q_grid_desc_m0_k_m1, q_grid_buf);

                // load KGrad Gemm A
                const auto sgrad_slice_idx =
                    Gemm2::ASrcBlockSliceWindowIterator::GetIndexTupleOfNumber(gemm2_loop_idx);
                constexpr auto mwave_range =
                    make_tuple(sgrad_slice_idx[I2],
                               sgrad_slice_idx[I2] +
                                   Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(I2));
                constexpr auto nwave_range =
                    make_tuple(sgrad_slice_idx[I3],
                               sgrad_slice_idx[I3] +
                                   Gemm2Params_N_O_M::ABlockSliceLengths_M0_N0_M1_N1::At(I3));

                if(gemm2_a_copy_subgroup.IsBelong(mwave_range, nwave_range))
                {
                    kgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds.Run(
                        Gemm2::a_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        make_tuple(
                            sgrad_slice_idx[I0], sgrad_slice_idx[I1], I0, I0, I0, I0, I0, I0),
                        sgrad_thread_buf,
                        Gemm2::a_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        gemm2_a_block_buf);
                }

                // kgrad slice window is moved with MoveSrcSliceWindow() since it is dynamic buffer
                // sgrad slice window is moved by loop index
                kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(q_grid_desc_m0_k_m1,
                                                                    Gemm2::b_block_slice_copy_step);

                block_sync_lds(); // sync before write
                kgrad_gemm_tile_q_blockwise_copy.RunWrite(Gemm2::b_block_desc_m0_o_m1,
                                                          gemm2_b_block_buf);

                block_sync_lds(); // sync before read
                v_slash_k_grad_blockwise_gemm.Run(
                    gemm2_a_block_buf, gemm2_b_block_buf, v_slash_k_grad_thread_buf);

            }); // end gemm dK
            // atomic_add dK
            kgrad_thread_copy_vgpr_to_global.Run(Gemm2::c_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                 v_slash_k_grad_thread_buf,
                                                 kgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 kgrad_grid_buf);

            // move slice window
            s_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                q_grid_desc_k0_m_k1,
                s_gemm_tile_a_block_reset_copy_step); // rewind K
            s_gemm_tile_k_blockwise_copy.MoveSrcSliceWindow(
                k_grid_desc_k0_n_k1,
                s_gemm_tile_b_block_reset_copy_step); // rewind K and step N
            vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_m0_o_m1,
                Gemm2::b_block_reset_copy_step); // rewind M
            vgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4, Gemm2::c_block_slice_copy_step); // step N
            pgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_o0_m_o1, pgrad_gemm_tile_ygrad_block_reset_copy_step); // rewind O
            pgrad_gemm_tile_v_blockwise_copy.MoveSrcSliceWindow(
                v_grid_desc_o0_n_o1,
                pgrad_gemm_tile_v_block_reset_copy_step); // rewind O and step N
            kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                q_grid_desc_m0_k_m1,
                Gemm2::b_block_reset_copy_step); // rewind M
            kgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                kgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4, Gemm2::c_block_slice_copy_step); // step N

            z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0, 0));

        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop

        // shuffle dQ and write
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              Gemm1NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                qgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                qgrad_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatCShuffle*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2)),                                    // M2 = MPerXdl
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2,                                      // N2 * N3 * N4 = NPerXdl
                        N3,
                        N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4>{}, Sequence<>{}, Sequence<1, 3, 5, 6, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                qgrad_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   SElementwiseOperation,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    scale_rp_dropout};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatCShuffle,        // typename SrcData,
                OutputDataType,       // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(qgrad_grid_desc_mblock_mperblock_kblock_kperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 qgrad_grid_desc_mblock_mperblock_kblock_kperblock,
                 make_multi_index(block_work_idx_m, 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, Gemm1NXdlPerWave, 1, 1, 1, N2, 1, N4>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           1,
                                           N2,
                                           1,
                                           N4>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, Gemm1NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              qgrad_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    qgrad_grid_desc_mblock_mperblock_kblock_kperblock,
                    qgrad_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        qgrad_grid_desc_mblock_mperblock_kblock_kperblock, c_global_step);
                }
            });
        }
    }
};

} // namespace ck
