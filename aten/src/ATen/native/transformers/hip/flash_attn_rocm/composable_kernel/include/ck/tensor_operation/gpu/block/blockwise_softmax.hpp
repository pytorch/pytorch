// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"

namespace ck {

template <index_t BlockSize,
          typename AccDataType,
          typename ThreadMap_M_K, // thread_id to m_k
          typename ThreadClusterDesc_M_K,
          typename ThreadSliceDesc_M_K,
          bool IgnoreNaN = false>
struct BlockwiseSoftmax
{
    static constexpr auto I0         = Number<0>{};
    static constexpr auto I1         = Number<1>{};
    static constexpr index_t MRepeat = ThreadSliceDesc_M_K{}.GetLength(I0);
    static constexpr index_t KRepeat = ThreadSliceDesc_M_K{}.GetLength(I1);

    using ThreadSliceDesc_M = decltype(
        make_naive_tensor_descriptor_packed(make_tuple(ThreadSliceDesc_M_K{}.GetLength(I0))));

    using ThreadwiseMaxReduce = typename conditional<
        IgnoreNaN,
        ThreadwiseReduction<AccDataType,
                            ThreadSliceDesc_M_K,
                            ThreadSliceDesc_M,
                            reduce::Max,
                            false,
                            detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>,
        ThreadwiseReduction<AccDataType,
                            ThreadSliceDesc_M_K,
                            ThreadSliceDesc_M,
                            reduce::Max,
                            false>>::type;

    using ThreadwiseSumReduce = typename conditional<
        IgnoreNaN,
        ThreadwiseReduction<AccDataType,
                            ThreadSliceDesc_M_K,
                            ThreadSliceDesc_M,
                            reduce::Add,
                            false,
                            detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>,
        ThreadwiseReduction<AccDataType,
                            ThreadSliceDesc_M_K,
                            ThreadSliceDesc_M,
                            reduce::Add,
                            false>>::type;

    using ThreadClusterLengths_M_K = decltype(ThreadClusterDesc_M_K{}.GetLengths());

    using BlockwiseMaxReduce = PartitionedBlockwiseReduction_v2<AccDataType,
                                                                BlockSize,
                                                                ThreadClusterLengths_M_K,
                                                                ThreadMap_M_K,
                                                                reduce::Max,
                                                                false>;

    using BlockwiseSumReduce = PartitionedBlockwiseReduction_v2<AccDataType,
                                                                BlockSize,
                                                                ThreadClusterLengths_M_K,
                                                                ThreadMap_M_K,
                                                                reduce::Add,
                                                                false>;

    using BufferType = StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MRepeat, true>;

    template <typename CThreadBuffer, typename WorkspaceBuffer>
    __host__ __device__ void Run(CThreadBuffer& in_thread_buf, WorkspaceBuffer& reduce_work_buf)
    {
        // find max value
        static_for<0, MRepeat, 1>{}([&](auto I) {
            max_value_buf(I) = reduce::Max::template GetIdentityValue<AccDataType>();
        });
        ThreadwiseMaxReduce::Reduce(in_thread_buf, max_value_buf);
        static_for<0, MRepeat, 1>{}([&](auto I) {
            BlockwiseMaxReduce::Reduce(reduce_work_buf, max_value_buf(I));
            block_sync_lds();
        });

        // calculate exp for elements, P=exp(s-max)
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) = IgnoreNaN && ck::math::isnan(in_thread_buf[offset])
                                            ? 0
                                            : math::exp(in_thread_buf[offset] - max_value_buf(iM));
            });
        });

        // sum data
        static_for<0, MRepeat, 1>{}([&](auto I) {
            sum_value_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
        });
        ThreadwiseSumReduce::Reduce(in_thread_buf, sum_value_buf);
        static_for<0, MRepeat, 1>{}([&](auto I) {
            BlockwiseSumReduce::Reduce(reduce_work_buf, sum_value_buf(I));
            block_sync_lds();
        });
    }

    template <typename CThreadBuffer, typename LSEBuffer>
    __host__ __device__ void RunWithPreCalcStats(CThreadBuffer& in_thread_buf,
                                                 const LSEBuffer& lse_thread_buf)
    {
        // calculate exp for elements using pre-calculated stats LSE (log-sum-exp)
        // Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
        //    = exp(Si) / exp(log(sum(exp() + ...)))
        //    = exp(Si - log(sum(exp() + ...)))
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) = IgnoreNaN && ck::math::isnan(in_thread_buf[offset])
                                            ? 0
                                            : math::exp(in_thread_buf[offset] - lse_thread_buf[iM]);
            });
        });
    }

    BufferType max_value_buf;
    BufferType sum_value_buf;
};

} // namespace ck
