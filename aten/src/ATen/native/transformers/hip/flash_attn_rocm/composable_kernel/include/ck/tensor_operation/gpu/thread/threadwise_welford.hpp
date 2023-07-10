// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/math_v2.hpp"

namespace ck {

// Assume
//  1) XDesc is known at compile-time
//  2) MeanVarDesc is known at compile-time
//  3) XBuffer is static buffer
//  4) MeanBuffer is static buffer
//  5) VarBuffer is static buffer
template <typename T, typename XThreadDesc_M_K, typename MeanVarThreadDesc_M>
struct ThreadwiseWelford
{
    static constexpr auto x_thread_desc_m_k      = XThreadDesc_M_K{};
    static constexpr auto mean_var_thread_desc_m = MeanVarThreadDesc_M{};

    static constexpr auto thread_x_length_m        = x_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto thread_x_length_k        = x_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto thread_mean_var_length_m = mean_var_thread_desc_m.GetLength(Number<0>{});

    static_assert(thread_x_length_m == thread_mean_var_length_m,
                  "lengths of source and mean/var buffer must match!");

    __device__ constexpr ThreadwiseWelford() : cur_count_(0), max_count_(0) {}

    __device__ inline void Update(T& mean, T& var, T x)
    {
        using ck::math::isnan;

        if(isnan(x))
        {
            mean = x;
            var  = x;
        }
        else
        {
            T delta = x - mean;
            mean += delta / cur_count_;
            T delta2 = x - mean;
            var += delta * delta2;
        }
    }

    template <typename XBufferType, typename MeanBufferType, typename VarBufferType>
    __device__ void
    Run(const XBufferType& x_buf_m_k, MeanBufferType& mean_buf_m, VarBufferType& var_buf_m)
    {
        // FIXME - Better naming for var_buf_m

        static_for<0, thread_x_length_k, 1>{}([&](auto iK) {
            if(cur_count_ < max_count_)
            {
                ++cur_count_;

                static_for<0, thread_x_length_m, 1>{}([&](auto iM) {
                    constexpr index_t out_offset =
                        mean_var_thread_desc_m.CalculateOffset(make_tuple(iM));

                    constexpr auto in_offset =
                        x_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                    Update(mean_buf_m(Number<out_offset>{}),
                           var_buf_m(Number<out_offset>{}),
                           x_buf_m_k[Number<in_offset>{}]);
                });
            }
        });
    };

    int cur_count_;
    int max_count_;
};

template <typename T,
          typename SrcMeanVarCountThreadDesc_M_K,
          typename DstMeanVarThreadDesc_M,
          bool GetActualVariance = false>
struct ThreadwiseWelfordMerge
{
    static constexpr auto src_thread_desc_m_k = SrcMeanVarCountThreadDesc_M_K{};
    static constexpr auto dst_thread_desc_m   = DstMeanVarThreadDesc_M{};

    static constexpr auto src_length_m = src_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto src_length_k = src_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto dst_length_m = dst_thread_desc_m.GetLength(Number<0>{});

    static_assert(src_length_m == dst_length_m, "lengths of source and dst buffer must match!");

    __device__ static void
    Merge(T& mean_a, T& var_a, int32_t& count_a, T mean_b, T var_b, int32_t count_b)
    {
        int count            = count_a + count_b;
        T count_b_over_count = count == 0 ? type_convert<T>(0) : type_convert<T>(count_b) / count;
        T delta              = mean_b - mean_a;
        mean_a += delta * count_b_over_count;
        var_a += var_b + delta * delta * count_a * count_b_over_count;
        count_a = count;
    }

    template <typename SrcMeanBufferType,
              typename SrcVarBufferType,
              typename SrcCountBufferType,
              typename DstMeanBufferType,
              typename DstVarBufferType,
              typename DstCountBufferType>
    __device__ static void Run(const SrcMeanBufferType& src_mean_buf,
                               const SrcVarBufferType& src_var_buf,
                               const SrcCountBufferType& src_count_buf,
                               DstMeanBufferType& dst_mean_buf,
                               DstVarBufferType& dst_var_buf,
                               DstCountBufferType& dst_count_buf)
    {
        static_for<0, src_length_m, 1>{}([&](auto iM) {
            static_for<0, src_length_k, 1>{}([&](auto iK) {
                constexpr auto src_offset = src_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                Merge(dst_mean_buf(iM),
                      dst_var_buf(iM),
                      dst_count_buf(iM),
                      src_mean_buf[Number<src_offset>{}],
                      src_var_buf[Number<src_offset>{}],
                      src_count_buf[Number<src_offset>{}]);
            });

            if constexpr(GetActualVariance)
            {
                dst_var_buf(iM) = dst_var_buf[iM] / dst_count_buf[iM];
            };
        });
    };
};

} // namespace ck
