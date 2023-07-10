// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

// clang-format off
// Assume:
//  1) work_buffer is buffer (typically LDS) allocated outside as workspace, does not include any in/out data
//  2) work_buffer has AccDataType elements, and space size is no less than BlockSize
//  3) in_out_value is the input data in vgpr from each thread
//  4) in_out_value is the over-written reduced output in vgpr for each thread
// clang-format on
template <typename AccDataType,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterArrangeOrder,
          typename OpReduce,
          bool PropagateNan,
          typename Accumulation =
              detail::AccumulateWithNanCheck<PropagateNan, OpReduce, AccDataType>>
struct PartitionedBlockwiseReduction
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static_assert(BufferLength_K > 1, "Parallel reduction need work on at least two elements");

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    template <typename BufferType>
    __device__ static void Reduce(BufferType& work_buffer, AccDataType& in_out_value)
    {
        static_assert(is_same<typename BufferType::type, AccDataType>{},
                      "Buffer data type should be consistent as AccDataType!");

        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        work_buffer(block_buf_desc_m_k.CalculateOffset(thread_cluster_idx)) = in_out_value;

        __syncthreads();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_k_cluster_id < indOffset)
            {
                index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                AccDataType opData1 = work_buffer[offset1];
                AccDataType opData2 = work_buffer[offset2];
                Accumulation::Calculate(opData1, opData2);
                work_buffer(offset1) = opData1;
            }

            __syncthreads();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        in_out_value = work_buffer[offset];
    };
};

// clang-format off
// Assume:
//  1) work_buffer is buffer (typically LDS) allocated outside as workspace, does not include any in/out data
//  2) work_buffer has AccDataType elements, and space size is no less than BlockSize
//  3) in_out_value is the input data in vgpr from each thread
//  4) in_out_value is the over-written reduced output in vgpr for each thread
// clang-format on
template <typename AccDataType,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterDesc,
          typename OpReduce,
          bool PropagateNan,
          typename Accumulation =
              detail::AccumulateWithNanCheck<PropagateNan, OpReduce, AccDataType>>
struct PartitionedBlockwiseReduction_v2
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static_assert(BufferLength_K > 1, "Parallel reduction need work on at least two elements");

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc = ThreadClusterDesc{};

    template <typename BufferType>
    __device__ static void Reduce(BufferType& work_buffer, AccDataType& in_out_value)
    {
        static_assert(is_same<typename BufferType::type, AccDataType>{},
                      "Buffer data type should be consistent as AccDataType!");

        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        work_buffer(block_buf_desc_m_k.CalculateOffset(thread_cluster_idx)) = in_out_value;

        __syncthreads();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_k_cluster_id < indOffset)
            {
                index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                AccDataType opData1 = work_buffer[offset1];
                AccDataType opData2 = work_buffer[offset2];
                Accumulation::Calculate(opData1, opData2);
                work_buffer(offset1) = opData1;
            }

            __syncthreads();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        in_out_value = work_buffer[offset];
    };
};

// clang-format off
// Assume:
//  1) work_val_buffer/work_idx_buffer is buffer (typically LDS) allocated outside as workspace, does not include any in/out data
//  2) work_val_buffer/work_idx_buffer has AccDataType/IndexDataType elements, and space size is no less than BlockSize
//  3) in_out_value/in_out_index is the input data in vgpr from each thread
//  4) in_out_value/in_out_index is the over-written reduced output in vgpr for each thread
// clang-format on
template <
    typename AccDataType,
    typename IndexDataType,
    index_t BlockSize,
    typename ThreadClusterLengths_M_K,
    typename ThreadClusterArrangeOrder,
    typename OpReduce,
    bool PropagateNan,
    typename Accumulation =
        detail::AccumulateWithIndexAndNanCheck<PropagateNan, OpReduce, AccDataType, IndexDataType>>
struct PartitionedBlockwiseReductionWithIndex
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static_assert(BufferLength_K > 1, "Parallel reduction need work on at least two elements");

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    // This interface accumulates on both data values and indices
    template <typename BufferType, typename IdxBufferType>
    __device__ static void Reduce(BufferType& work_val_buffer,
                                  IdxBufferType& work_idx_buffer,
                                  AccDataType& in_out_value,
                                  IndexDataType& in_out_index)
    {
        static_assert(is_same<typename BufferType::type, AccDataType>{},
                      "Buffer data type should be consistent as AccDataType!");
        static_assert(is_same<typename IdxBufferType::type, IndexDataType>{},
                      "Buffer data type should be consistent as IndexDataType!");

        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        work_val_buffer(block_buf_desc_m_k.CalculateOffset(thread_cluster_idx)) = in_out_value;
        work_idx_buffer(block_buf_desc_m_k.CalculateOffset(thread_cluster_idx)) = in_out_index;

        __syncthreads();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << I();

            if(thread_k_cluster_id % (indOffset * 2) == 0)
            {
                index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                AccDataType opData1      = work_val_buffer[offset1];
                AccDataType opData2      = work_val_buffer[offset2];
                IndexDataType currIndex1 = work_idx_buffer[offset1];
                IndexDataType currIndex2 = work_idx_buffer[offset2];

                Accumulation::Calculate(opData1, opData2, currIndex1, currIndex2);
                work_val_buffer(offset1) = opData1;
                work_idx_buffer(offset1) = currIndex1;
            }

            __syncthreads();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        in_out_value = work_val_buffer[offset];
        in_out_index = work_idx_buffer[offset];
    };
};

} // namespace ck
