// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

// Assume
//  1) SrcDesc is known at compile-time
//  2) DstDesc is known at compile-time
//  3) SrcBuffer is static buffer
//  4) DstBuffer is static buffer
template <typename AccDataType,
          typename SrcThreadDesc_M_K,
          typename DstThreadDesc_M,
          typename OpReduce,
          bool PropagateNan,
          typename Accumulation =
              detail::AccumulateWithNanCheck<PropagateNan, OpReduce, AccDataType>>
struct ThreadwiseReduction
{
    static constexpr auto src_thread_desc_m_k = SrcThreadDesc_M_K{};
    static constexpr auto dst_thread_desc_m   = DstThreadDesc_M{};

    static constexpr auto src_length_m = src_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto src_length_k = src_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto dst_length_m = dst_thread_desc_m.GetLength(Number<0>{});

    static_assert(src_length_m == dst_length_m, "lengths of source and dst buffer must match!");

    using Op = OpReduce;

    template <typename SrcBufferType, typename DstBufferType>
    __device__ static void Reduce(const SrcBufferType& src_buf, DstBufferType& dst_buf)
    {
        static_for<0, src_length_m, 1>{}([&](auto iM) {
            constexpr index_t out_offset = dst_thread_desc_m.CalculateOffset(make_tuple(iM));

            static_for<0, src_length_k, 1>{}([&](auto iK) {
                constexpr auto offset = src_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                Accumulation::Calculate(dst_buf(Number<out_offset>{}), src_buf[Number<offset>{}]);
            });
        });
    };
};

// Assume
//  1) SrcDesc is known at compile-time
//  2) DstDesc is known at compile-time
//  3) SrcBuffer is static buffer
//  4) DstBuffer is static buffer
template <
    typename AccDataType,
    typename IndexDataType,
    typename SrcThreadDesc_M_K,
    typename DstThreadDesc_M,
    typename OpReduce,
    bool PropagateNan,
    typename Accumulation =
        detail::AccumulateWithIndexAndNanCheck<PropagateNan, OpReduce, AccDataType, IndexDataType>>
struct ThreadwiseReductionWithIndex
{
    static constexpr auto src_thread_desc_m_k = SrcThreadDesc_M_K{};
    static constexpr auto dst_thread_desc_m   = DstThreadDesc_M{};

    static constexpr auto src_length_m = src_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto src_length_k = src_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto dst_length_m = dst_thread_desc_m.GetLength(Number<0>{});

    static_assert(src_length_m == dst_length_m, "lengths of source and dst buffer must match!");

    template <typename SrcValueBufferType,
              typename SrcIndexBufferType,
              typename DstValueBufferType,
              typename DstIndexBufferType>
    __device__ static void Reduce(const SrcValueBufferType& src_val_buf,
                                  const SrcIndexBufferType& src_idx_buf,
                                  DstValueBufferType& dst_val_buf,
                                  DstIndexBufferType& dst_idx_buf)
    {
        static_for<0, src_length_m, 1>{}([&](auto iM) {
            constexpr index_t out_offset = dst_thread_desc_m.CalculateOffset(make_tuple(iM));

            static_for<0, src_length_k, 1>{}([&](auto iK) {
                constexpr auto offset = src_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                Accumulation::Calculate(dst_val_buf(Number<out_offset>{}),
                                        src_val_buf[Number<offset>{}],
                                        dst_idx_buf(Number<out_offset>{}),
                                        src_idx_buf[Number<offset>{}]);
            });
        });
    };
};

} // namespace ck
