/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <ATen/native/transformers/cuda/flash_attn/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pytorch_flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const &S,
                                               Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &Cos,
                                               Tensor<Engine2, Layout2> const &Sin,
                                               Tensor<Engine3, Layout3> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        cute::copy(Cos(_, m, k), rCos(_, m, k));
                        cute::copy(Sin(_, m, k), rSin(_, m, k));
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS) / 2; ++i) {
                            float real = S_fp32(2 * i) * cos_fp32(i) - S_fp32(2 * i + 1) * sin_fp32(i);
                            float imag = S_fp32(2 * i) * sin_fp32(i) + S_fp32(2 * i + 1) * cos_fp32(i);
                            S_fp32(2 * i) = real;
                            S_fp32(2 * i + 1) = imag;
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        cute::copy(S_og_type, rS(_, m, k));
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim / 2;
                        Tensor gS_other = make_tensor(S(_, m, k).data() + (is_left ? rotary_dim / 2 : -rotary_dim / 2), S(_, m, k).layout());
                        cute::copy(gS_other, rS_other);
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); print_tensor(rS_other); }
                        Tensor gCos = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Cos(_, m, k).layout());
                        Tensor gSin = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Sin(_, m, k).layout());
                        cute::copy(gCos, rCos(_, m, k));
                        cute::copy(gSin, rSin(_, m, k));
                        // if (cute::thread0()) { print_tensor(rCos(_, m, k)); print_tensor(rSin(_, m, k)); }
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor S_other_fp32 = convert_type<float>(rS_other);
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS); ++i) {
                            S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        cute::copy(S_og_type, rS(_, m, k));
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); }
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace pytorch_flash
