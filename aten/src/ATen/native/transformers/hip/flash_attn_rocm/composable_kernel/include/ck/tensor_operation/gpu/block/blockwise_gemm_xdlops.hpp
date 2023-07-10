// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

enum struct LoopScheduler
{
    Default,
    Interwave,
};

constexpr LoopScheduler make_default_loop_scheduler()
{
#if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
    return LoopScheduler::Interwave;
#else
    return LoopScheduler::Default;
#endif // if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
}

template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
__host__ __device__ static constexpr auto
MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K(const TileDesc_K0_MN_K1&)
{
    constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
    constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

    return transform_tensor_descriptor(
        TileDesc_K0_MN_K1{},
        make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                   make_unmerge_transform(
                       make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
        make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool TransposeC = false>
struct BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = get_warp_size();

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);
    static constexpr index_t KPerBlock =
        BK0NK1BlockDesc{}.GetLength(I0) * BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr index_t A_K0 = AK0MK1BlockDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BK0NK1BlockDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = AK0MK1BlockDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, TransposeC>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], KPerThread * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, xdlops_b_idx[I1], KPerThread * xdlops_b_idx[I0]);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk4D(xdlops_i, blk_i);

        return make_tuple(Number<m0>{},
                          Number<n0>{},
                          waveId_m,
                          waveId_n,
                          blk_idx[I0],
                          blk_idx[I1],
                          blk_idx[I2],
                          blk_idx[I3]);
    }

    __host__ __device__ BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1()
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, N, M0, M1, M2));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto MakeABlockDescriptor_M0_M1_M2_K()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(
                make_merge_transform_v3_division_mod(make_tuple(Number<A_K0>{}, Number<A_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerXDL>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor_N0_N1_N2_K()
    {
        return transform_tensor_descriptor(
            BK0NK1BlockDesc{},
            make_tuple(
                make_merge_transform_v3_division_mod(make_tuple(Number<B_K0>{}, Number<B_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_k = MakeABlockDescriptor_M0_M1_M2_K();
    static constexpr auto b_block_desc_n0_n1_n2_k = MakeBBlockDescriptor_N0_N1_N2_K();

    __host__ __device__ static constexpr auto MakeCThreadTileIterator()
    {
        constexpr auto c_thread_lengths = conditional_expr<TransposeC>(
            GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths(),
            GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths());
        return SpaceFillingCurve<
            decltype(c_thread_lengths),
            typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
            typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
            false>{}; // SnakeCurved
    }

    __host__ __device__ static constexpr auto MakeCThreadIndexAdaptor8DTo2D()
    {
        if constexpr(TransposeC)
        {
            constexpr auto c_thread_desc = GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            constexpr auto m0            = c_thread_desc.GetLength(Number<0>{});
            constexpr auto n0            = c_thread_desc.GetLength(Number<1>{});
            constexpr auto m1            = c_thread_desc.GetLength(Number<2>{});
            constexpr auto n1            = c_thread_desc.GetLength(Number<3>{});
            constexpr auto m2            = c_thread_desc.GetLength(Number<4>{});
            constexpr auto n2            = c_thread_desc.GetLength(Number<5>{});
            constexpr auto n3            = c_thread_desc.GetLength(Number<6>{});
            constexpr auto n4            = c_thread_desc.GetLength(Number<7>{});
            constexpr auto thread_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(m0, m1, m2)),
                           make_unmerge_transform(make_tuple(n0, n1, n2, n3, n4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
            return thread_idx_to_m_n_adaptor;
        }
        else
        {
            constexpr auto c_thread_desc = GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
            constexpr auto m0            = c_thread_desc.GetLength(Number<0>{});
            constexpr auto n0            = c_thread_desc.GetLength(Number<1>{});
            constexpr auto m1            = c_thread_desc.GetLength(Number<2>{});
            constexpr auto n1            = c_thread_desc.GetLength(Number<3>{});
            constexpr auto m2            = c_thread_desc.GetLength(Number<4>{});
            constexpr auto m3            = c_thread_desc.GetLength(Number<5>{});
            constexpr auto m4            = c_thread_desc.GetLength(Number<6>{});
            constexpr auto n2            = c_thread_desc.GetLength(Number<7>{});
            constexpr auto thread_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(m0, m1, m2, m3, m4)),
                           make_unmerge_transform(make_tuple(n0, n1, n2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));
            return thread_idx_to_m_n_adaptor;
        }
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            // read A
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                               make_tuple(m0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read B
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf);

                static_for<0, KPerThread, KPack>{}([&](auto k) {
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<FloatAB>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, k + i))>{}];
                        b_thread_vec.template AsType<FloatAB>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, k + i))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template Run(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
            });
        });
    }

    protected:
    // A[M0, M1, M2, KPerThread]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPerThread>{}));

    // B[N0, N1, N2, KPerThread]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPerThread>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

// Note: To facilitate the inter-wave loop scheduler, we need to explicitly set the macro
// CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=1 as a few intrinsics are not yet available in
// the latest ROCm release. For unsupported compilers, inter-wave loop scheduler falls back to the
// default loop scheduler which is given by the macro CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=0
template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          index_t NumMacClusters = CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS>
struct BlockwiseGemmXdlopsInterwave_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
    : public BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                 FloatAB,
                                                                 FloatAcc,
                                                                 AK0MK1BlockDesc,
                                                                 BK0NK1BlockDesc,
                                                                 MPerXDL,
                                                                 NPerXDL,
                                                                 MRepeat,
                                                                 NRepeat,
                                                                 KPack>
{
    using Base = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                     FloatAB,
                                                                     FloatAcc,
                                                                     AK0MK1BlockDesc,
                                                                     BK0NK1BlockDesc,
                                                                     MPerXDL,
                                                                     NPerXDL,
                                                                     MRepeat,
                                                                     NRepeat,
                                                                     KPack>;

#if CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING
    using Base::a_block_desc_m0_m1_m2_k;
    using Base::A_K1;
    using Base::b_block_desc_n0_n1_n2_k;
    using Base::B_K1;
    using Base::c_thread_buf_;
    using Base::c_thread_desc_;
    using Base::CalculateAThreadOriginDataIndex;
    using Base::CalculateBThreadOriginDataIndex;
    using Base::I0;
    using Base::I1;
    using Base::KPerThread;
    using Base::xdlops_gemm;

    static constexpr index_t KPerInnerLoop = math::max(KPerThread / NumMacClusters, KPack);

    // 2-wave optimized blockwise gemm
    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, KPerThread, KPerInnerLoop>{}([&](auto k) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                // read A
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, k),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(m0, I0, I0, I0),
                                   a_thread_buf);
            });
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read B
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, k),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(n0, I0, I0, I0),
                                   b_thread_buf);
            });
            __builtin_amdgcn_sched_barrier(0);
            // NOTE: Synchronize threads in a workgroup at the start of each MAC cluster, but except
            // the first, as we can shorten non-MAC cluster a bit and there's no observable negative
            // impact. The desired effect is waves in a workgroup executing MAC in sync. This avoids
            // some out-of-sync waves hijacking MAC resource from other workgroups and reducing the
            // chance of latency hiding by waiting for the rest of the workgroup at the eventual
            // sync point.
            if constexpr(k.value != 0 || KPerInnerLoop == KPerThread)
            {
                asm volatile("s_barrier" ::);
                __builtin_amdgcn_sched_barrier(0);
            }
            static_for<0, KPerInnerLoop, KPack>{}([&](auto k_) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto i) {
                            a_thread_vec.template AsType<FloatAB>()(i) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, 0, 0, k_ + i))>{}];
                            b_thread_vec.template AsType<FloatAB>()(i) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, 0, 0, k_ + i))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        // The block_sync_lds() here performs double duty:
                        // A) safeguard against data hazard because barrier from blockwise_gemm is
                        // moved here B) reduce VMEM FIFO congestion by applying small delays to
                        // different wavefronts It is performed near the end of MAC cluster to
                        // minimize lgkmcnt penalty
                        if constexpr(k.value == KPerThread - KPerInnerLoop &&
                                     k_.value == KPerInnerLoop - KPack && m0.value == MRepeat - 1 &&
                                     n0.value == NRepeat - 1)
                        {
                            __builtin_amdgcn_sched_barrier(0);
                            block_sync_lds();
                            __builtin_amdgcn_sched_barrier(0);
                        }

                        // TODO: insert setprio in more precise manner since we
                        // could have more than >1 MFMA instructions in single call
                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        if constexpr(k_.value == 0 && m0.value == 0 && n0.value == 0)
                        {
                            __builtin_amdgcn_sched_barrier(0);
                            __builtin_amdgcn_s_setprio(1);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    });
                });
            });
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
        });
    }

    protected:
    // A[M0, M1, M2, KPerInnerLoop]
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, I1, I1, Number<KPerInnerLoop>{}));

    // B[N0, N1, N2, KPerInnerLoop]
    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat>{}, I1, I1, Number<KPerInnerLoop>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPerInnerLoop>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPerInnerLoop>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};

#endif // #if CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          LoopScheduler LoopSched>
constexpr auto BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                   FloatAB,
                                                                   FloatAcc,
                                                                   AK0MK1BlockDesc,
                                                                   BK0NK1BlockDesc,
                                                                   MPerXDL,
                                                                   NPerXDL,
                                                                   MRepeat,
                                                                   NRepeat,
                                                                   KPack>{};
    }
    else if constexpr(LoopSched == LoopScheduler::Interwave)
    {
        return BlockwiseGemmXdlopsInterwave_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                            FloatAB,
                                                                            FloatAcc,
                                                                            AK0MK1BlockDesc,
                                                                            BK0NK1BlockDesc,
                                                                            MPerXDL,
                                                                            NPerXDL,
                                                                            MRepeat,
                                                                            NRepeat,
                                                                            KPack>{};
    }
};

// Blockwise gemm supporting
// 1. regular XDL output M2_M3_M4_M2 and transposed XDL output M2_N2_N3_N4
// 2. decoupled input tile descriptor and mma tile descriptor in order to support both vgpr and LDS
// source buffer
// 3. configurable k index starting position and step size after each FMA/XDL instruction
template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool TransposeC = false,
          index_t AMmaKStride =
              KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, TransposeC>{}.K0PerXdlops,
          index_t BMmaKStride =
              KPack* XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, TransposeC>{}.K0PerXdlops>
struct BlockwiseGemmXdlops_v2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = get_warp_size();

    static constexpr index_t A_K0 = ATileDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BTileDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = ATileDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BTileDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, TransposeC>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);

    static_assert(KPerThread % KPack == 0,
                  "Wrong KPack setting; try increasing KPerThread or decreasing KPack");

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], KPack * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, xdlops_b_idx[I1], KPack * xdlops_b_idx[I0]);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk4D(xdlops_i, blk_i);

        return make_tuple(
            m0, n0, waveId_m, waveId_n, blk_idx[I0], blk_idx[I1], blk_idx[I2], blk_idx[I3]);
    }

    using Tuple4 = decltype(CalculateAThreadOriginDataIndex());

    __host__ __device__ BlockwiseGemmXdlops_v2(Tuple4 a_origin = CalculateAThreadOriginDataIndex(),
                                               Tuple4 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AMmaTileDesc::IsKnownAtCompileTime() && BMmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ BlockwiseGemmXdlops_v2(const BlockwiseGemmXdlops_v2& other)
        : a_thread_copy_(other.a_origin), b_thread_copy_(other.b_origin)
    {
    }

    __device__ void SetABlockStartWindow(Tuple4 a_origin = CalculateAThreadOriginDataIndex())
    {
        a_thread_copy_.SetSrcCoord(a_origin);
    }

    __device__ void SetBBlockStartWindow(Tuple4 b_origin = CalculateBThreadOriginDataIndex())
    {
        b_thread_copy_.SetSrcCoord(b_origin);
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, N, M0, M1, M2));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    static constexpr AMmaTileDesc a_block_desc_m0_m1_m2_k;
    static constexpr BMmaTileDesc b_block_desc_n0_n1_n2_k;

    __host__ __device__ static constexpr auto MakeCThreadTileIterator()
    {
        constexpr auto c_thread_lengths = conditional_expr<TransposeC>(
            GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths(),
            GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths());
        return SpaceFillingCurve<
            decltype(c_thread_lengths),
            typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
            typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
            false>{}; // SnakeCurved
    }

    __host__ __device__ static constexpr auto MakeCThreadIndexAdaptor8DTo2D()
    {
        if constexpr(TransposeC)
        {
            constexpr auto c_thread_desc = GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            constexpr auto m0            = c_thread_desc.GetLength(Number<0>{});
            constexpr auto n0            = c_thread_desc.GetLength(Number<1>{});
            constexpr auto m1            = c_thread_desc.GetLength(Number<2>{});
            constexpr auto n1            = c_thread_desc.GetLength(Number<3>{});
            constexpr auto m2            = c_thread_desc.GetLength(Number<4>{});
            constexpr auto n2            = c_thread_desc.GetLength(Number<5>{});
            constexpr auto n3            = c_thread_desc.GetLength(Number<6>{});
            constexpr auto n4            = c_thread_desc.GetLength(Number<7>{});
            constexpr auto thread_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(m0, m1, m2)),
                           make_unmerge_transform(make_tuple(n0, n1, n2, n3, n4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
            return thread_idx_to_m_n_adaptor;
        }
        else
        {
            constexpr auto c_thread_desc = GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
            constexpr auto m0            = c_thread_desc.GetLength(Number<0>{});
            constexpr auto n0            = c_thread_desc.GetLength(Number<1>{});
            constexpr auto m1            = c_thread_desc.GetLength(Number<2>{});
            constexpr auto n1            = c_thread_desc.GetLength(Number<3>{});
            constexpr auto m2            = c_thread_desc.GetLength(Number<4>{});
            constexpr auto m3            = c_thread_desc.GetLength(Number<5>{});
            constexpr auto m4            = c_thread_desc.GetLength(Number<6>{});
            constexpr auto n2            = c_thread_desc.GetLength(Number<7>{});
            constexpr auto thread_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(m0, m1, m2, m3, m4)),
                           make_unmerge_transform(make_tuple(n0, n1, n2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));
            return thread_idx_to_m_n_adaptor;
        }
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, KPerThread / KPack, 1>{}([&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                // read A
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   a_thread_buf);

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    // read B
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(I0, I0, I0, I0),
                                       b_thread_buf);
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<FloatAB>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, i))>{}];
                        b_thread_vec.template AsType<FloatAB>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, i))>{}];
                    });

                    using mfma_input_type =
                        typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template Run(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
            });
        });
    }

    protected:
    // A[M0, M1, M2, KPack]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPack>{}));

    // B[N0, N1, N2, KPack]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPack>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
