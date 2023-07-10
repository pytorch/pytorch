// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. AThreadDesc_E1_K_E2, BThreadDesc_E1_N_Ho_Wo_E2, CThreadDesc_K_N_Ho_Wo are known at
//   compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_E1_K_E2,
          typename BThreadDesc_E1_N_Ho_Wo_E2,
          typename CThreadDesc_K_N_Ho_Wo,
          typename enable_if<AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                                 BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                                 CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseGemmDlops_km_kn_mn_v3
{

    template <typename ABuffer,
              typename AOriginIdx,
              typename BBuffer,
              typename BOriginIdx,
              typename CBuffer,
              typename COriginIdx>
    __device__ static void Run(const ABuffer& a_buf,
                               AOriginIdx,
                               const BBuffer& b_buf,
                               BOriginIdx,
                               CBuffer& c_buf,
                               COriginIdx)
    {

        static_assert(AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                          BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
                      "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename ABuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto E1 = AThreadDesc_E1_K_E2{}.GetLength(I0);
        constexpr auto K  = AThreadDesc_E1_K_E2{}.GetLength(I1);
        constexpr auto E2 = AThreadDesc_E1_K_E2{}.GetLength(I2);

        constexpr auto Ho = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
        constexpr auto Wo = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        if constexpr((Ho % 2 == 0) && (Wo % 2 == 0))
        {
            constexpr auto SubHW = 2;

            static_for<0, K, 1>{}([&](auto k) {
                static_for<0, Ho, SubHW>{}([&](auto h) {
                    static_for<0, Wo, SubHW>{}([&](auto w) {
                        static_for<0, E1, 1>{}([&](auto e1) {
                            static_for<0, E2, 1>{}([&](auto e2) {
                                constexpr index_t a_offset = AThreadDesc_E1_K_E2{}.CalculateOffset(
                                    a_origin_idx + make_tuple(e1, k, e2));

                                constexpr index_t b0_offset =
                                    BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                        b_origin_idx + make_tuple(e1, 0, h, w, e2));

                                constexpr index_t b1_offset =
                                    BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                        b_origin_idx + make_tuple(e1, 0, h, w + 1, e2));

                                constexpr index_t b2_offset =
                                    BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                        b_origin_idx + make_tuple(e1, 0, h + 1, w, e2));

                                constexpr index_t b3_offset =
                                    BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                        b_origin_idx + make_tuple(e1, 0, h + 1, w + 1, e2));

                                constexpr index_t c0_offset =
                                    CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx +
                                                                            make_tuple(k, 0, h, w));

                                constexpr index_t c1_offset =
                                    CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                                        c_origin_idx + make_tuple(k, 0, h, w + 1));

                                constexpr index_t c2_offset =
                                    CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                                        c_origin_idx + make_tuple(k, 0, h + 1, w));

                                constexpr index_t c3_offset =
                                    CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                                        c_origin_idx + make_tuple(k, 0, h + 1, w + 1));

                                amd_assembly_outer_product_1x4(a_buf[Number<a_offset>{}],
                                                               b_buf[Number<b0_offset>{}],
                                                               b_buf[Number<b1_offset>{}],
                                                               b_buf[Number<b2_offset>{}],
                                                               b_buf[Number<b3_offset>{}],
                                                               c_buf(Number<c0_offset>{}),
                                                               c_buf(Number<c1_offset>{}),
                                                               c_buf(Number<c2_offset>{}),
                                                               c_buf(Number<c3_offset>{}));
                            });
                        });
                    });
                });
            });
        }
        else
        {

            static_for<0, K, 1>{}([&](auto k) {
                static_for<0, Ho, 1>{}([&](auto h) {
                    static_for<0, Wo, 1>{}([&](auto w) {
                        static_for<0, E1, 1>{}([&](auto e1) {
                            static_for<0, E2, 1>{}([&](auto e2) {
                                constexpr index_t a_offset = AThreadDesc_E1_K_E2{}.CalculateOffset(
                                    a_origin_idx + make_tuple(e1, k, e2));

                                constexpr index_t b_offset =
                                    BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                        b_origin_idx + make_tuple(e1, 0, h, w, e2));

                                constexpr index_t c_offset =
                                    CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx +
                                                                            make_tuple(k, 0, h, w));

                                inner_product<FloatA, FloatB, FloatC>(a_buf[Number<a_offset>{}],
                                                                      b_buf[Number<b_offset>{}],
                                                                      c_buf(Number<c_offset>{}));
                            });
                        });
                    });
                });
            });
        }
    }
};

} // namespace ck
#endif
