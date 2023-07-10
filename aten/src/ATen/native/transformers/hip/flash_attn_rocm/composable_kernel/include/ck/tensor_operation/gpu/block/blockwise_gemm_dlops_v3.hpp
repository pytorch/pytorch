// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_BLOCKWISE_GEMM_DLOPS_V3_HPP
#define CK_BLOCKWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "threadwise_gemm_dlops_v3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_E1_K1_E2,
          typename BBlockDesc_E1_N_Ho_Wo_E2,
          typename CThreadDesc_K_N_Ho_Wo,
          index_t EPerThreadLoop,
          index_t KPerThreadLoop>
struct BlockwiseGemmDlops_km_kn_m0m1n0n1_v3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    using AIndex = MultiIndex<3>;
    using BIndex = MultiIndex<3>;
    using CIndex = MultiIndex<4>;

    static constexpr auto E1        = ABlockDesc_E1_K1_E2{}.GetLength(I0);
    static constexpr auto KPerBlock = ABlockDesc_E1_K1_E2{}.GetLength(I1);
    static constexpr auto E2        = ABlockDesc_E1_K1_E2{}.GetLength(I2);

    static constexpr auto HoPerBlock = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
    static constexpr auto WoPerBlock = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

    static constexpr auto KPerThread  = CThreadDesc_K_N_Ho_Wo{}.GetLength(I0);
    static constexpr auto HoPerThread = CThreadDesc_K_N_Ho_Wo{}.GetLength(I2);
    static constexpr auto WoPerThread = CThreadDesc_K_N_Ho_Wo{}.GetLength(I3);

    static constexpr auto a_thread_mtx_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<EPerThreadLoop>{}, Number<KPerThreadLoop>{}, Number<E2>{}));

    static constexpr auto b_thread_mtx_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<EPerThreadLoop>{},
                                                       Number<1>{},
                                                       Number<HoPerThread>{},
                                                       Number<WoPerThread>{},
                                                       Number<E2>{}));

    static constexpr auto c_thread_mtx_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KPerThreadLoop>{}, Number<1>{}, Number<HoPerThread>{}, Number<WoPerThread>{}));

    __device__ BlockwiseGemmDlops_km_kn_m0m1n0n1_v3()
        : c_thread_origin_data_idx_{GetBeginOfCThreadDesc_K_N_Ho_Wo(get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_origin_data_idx_[I0] * KPerThread, 0)}
    {
        static_assert(ABlockDesc_E1_K1_E2::IsKnownAtCompileTime() &&
                          BBlockDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(
            ABlockDesc_E1_K1_E2{}.GetLength(I0) == BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I0) &&
                ABlockDesc_E1_K1_E2{}.GetLength(I2) == BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I4),
            "wrong! E dimension not consistent\n");

        static_assert(E1 % EPerThreadLoop == 0, "");
        static_assert(KPerThread % KPerThreadLoop == 0, "");

        static_assert(KPerBlock % KPerThread == 0 && HoPerBlock % HoPerThread == 0 &&
                          WoPerBlock % WoPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        constexpr auto KThreadCluster = KPerBlock / KPerThread;
        constexpr auto HThreadCluster = HoPerBlock / HoPerThread;
        constexpr auto WThreadCluster = WoPerBlock / WoPerThread;

        static_assert(BlockSize == KThreadCluster * HThreadCluster * WThreadCluster,
                      "wrong! wrong blocksize\n");
    }

    __device__ static constexpr auto GetCThreadDesc_K_N_Ho_WoLengths()
    {
        return Sequence<KPerThread, I1, HoPerThread, WoPerThread>{};
    }

    __device__ static CIndex GetBeginOfCThreadDesc_K_N_Ho_Wo(index_t thread_id)
    {
        constexpr auto K0 = KPerBlock / KPerThread;
        constexpr auto N0 = I1;
        constexpr auto H0 = HoPerBlock / HoPerThread;
        constexpr auto W0 = WoPerBlock / WoPerThread;

        constexpr auto c_threadid_to_k_n_h_w_thread_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(K0, N0, H0, W0))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto c_k_n_h_w_thread_cluster_idx =
            c_threadid_to_k_n_h_w_thread_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(thread_id));

        return c_k_n_h_w_thread_cluster_idx;
    }

    template <typename ABlockBuffer, typename BThreadBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BThreadBuffer& b_thread_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(
            is_same<remove_cvref_t<typename ABlockBuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BThreadBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CThreadBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto a_block_mtx = ABlockDesc_E1_K1_E2{};

        // thread A buffer for GEMM
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatA, a_thread_mtx_.GetElementSpaceSize(), true>
            a_thread_buf;

        constexpr auto threadwise_gemm = ThreadwiseGemmDlops_km_kn_mn_v3<FloatA,
                                                                         FloatB,
                                                                         FloatC,
                                                                         decltype(a_thread_mtx_),
                                                                         decltype(b_thread_mtx_),
                                                                         decltype(c_thread_mtx_)>{};

        static_for<0, E1, EPerThreadLoop>{}([&](auto e_begin) {
            static_for<0, KPerThread, KPerThreadLoop>{}([&](auto k_begin) {
                a_thread_copy_.Run(a_block_mtx,
                                   make_tuple(e_begin, k_begin, I0),
                                   a_block_buf,
                                   a_thread_mtx_,
                                   make_tuple(I0, I0, I0),
                                   a_thread_buf);

                threadwise_gemm.Run(a_thread_buf,
                                    make_tuple(I0, I0, I0),
                                    b_thread_buf,
                                    make_tuple(e_begin, I0, I0, I0, I0),
                                    c_thread_buf,
                                    make_tuple(k_begin, I0, I0, I0));
            });
        });
    }

    template <typename ABlockSliceMoveStepIdx>
    __device__ void MoveABlockSliceWindow(const ABlockSliceMoveStepIdx& a_block_slice_move_step_idx)
    {
        a_thread_copy_.MoveSrcSliceWindow(ABlockDesc_E1_K1_E2{}, a_block_slice_move_step_idx);
    }

    private:
    using AThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<FloatA,
                                         FloatA,
                                         ABlockDesc_E1_K1_E2,
                                         decltype(a_thread_mtx_),
                                         Sequence<EPerThreadLoop, KPerThreadLoop, E2>,
                                         Sequence<0, 1, 2>,
                                         2,
                                         E2,
                                         E2>;

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
};

} // namespace ck
#endif
