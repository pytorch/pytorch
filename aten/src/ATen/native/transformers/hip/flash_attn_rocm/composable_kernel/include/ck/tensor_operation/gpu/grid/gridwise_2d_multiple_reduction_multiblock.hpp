// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseMultipleReduction,
          index_t NumReduction,
          typename InDataType,
          typename OutDataTypePointerTuple,
          typename AccDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M_Tuple,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple>
__global__ void
kernel_multiple_reduce_multiblock(const InGridDesc_M_K in_grid_desc_m_k,
                                  const OutGridDesc_M_Tuple out_grid_desc_m_tuple,
                                  const InElementwiseOperationTuple in_elementwise_op_tuple,
                                  const AccElementwiseOperationTuple acc_elementwise_op_tuple,
                                  index_t block_group_size,
                                  index_t num_k_block_tile_iteration,
                                  Array<AccDataType, NumReduction> alpha_values,
                                  const InDataType* const __restrict__ p_in_value_global,
                                  Array<AccDataType, NumReduction> beta_values,
                                  OutDataTypePointerTuple p_out_value_global_tuple)
{
    GridwiseMultipleReduction::Run(in_grid_desc_m_k,
                                   out_grid_desc_m_tuple,
                                   in_elementwise_op_tuple,
                                   acc_elementwise_op_tuple,
                                   block_group_size,
                                   num_k_block_tile_iteration,
                                   alpha_values,
                                   p_in_value_global,
                                   beta_values,
                                   p_out_value_global_tuple);
};

template <index_t NumReduction,
          typename InDataType,
          typename OutDataTypePointerTuple,
          typename AccDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M_Tuple,
          typename ReduceOperation,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          typename OutDstVectorSizeSeq>
struct GridwiseMultipleReduction_mk_to_m_multiblock
{
    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(NumReduction == OutDataTypePointerTuple::Size() &&
                      NumReduction == OutGridDesc_M_Tuple::Size() &&
                      NumReduction == OutDstVectorSizeSeq::Size() &&
                      NumReduction == InElementwiseOperationTuple::Size() &&
                      NumReduction == AccElementwiseOperationTuple::Size(),
                  "All tuple should have the same size as the number of Reductions!");

    static constexpr bool reorder_thread_cluster = (InSrcVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using BlockwiseReduce = PartitionedBlockwiseReduction<AccDataType,
                                                          BlockSize,
                                                          ThreadClusterLengths_M_K,
                                                          ThreadClusterArrangeOrder,
                                                          ReduceOperation,
                                                          PropagateNan>;

    using ThreadwiseReduce = ThreadwiseReduction<AccDataType,
                                                 ThreadReduceSrcDesc_M_K,
                                                 ThreadReduceDstDesc_M,
                                                 ReduceOperation,
                                                 PropagateNan>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    using Accumulation = detail::AccumulateWithNanCheck<PropagateNan, ReduceOperation, AccDataType>;

    __device__ static void Run(const InGridDesc_M_K& in_grid_desc_m_k,
                               const OutGridDesc_M_Tuple& out_grid_desc_m_tuple,
                               const InElementwiseOperationTuple& in_elementwise_op_tuple,
                               const AccElementwiseOperationTuple& acc_elementwise_op_tuple,
                               index_t block_group_size,
                               index_t num_k_block_tile_iteration,
                               Array<AccDataType, NumReduction> alpha_values,
                               const InDataType* const __restrict__ p_in_value_global,
                               Array<AccDataType, NumReduction> beta_values,
                               OutDataTypePointerTuple p_out_value_global_tuple)
    {
        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        // LDS,  reused by all reductions
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        auto out_global_val_buf_tuple = generate_tuple(
            [&](auto iR) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_value_global_tuple[iR], out_grid_desc_m_tuple[iR].GetElementSpaceSize());
            },
            Number<NumReduction>{});

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_buf;

        auto in_thread_buf_tuple = generate_tuple(
            [&](auto iR) {
                (void)iR;
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * KThreadSliceSize,
                                    true>{};
            },
            Number<NumReduction>{});

        auto accu_value_buf_tuple = generate_tuple(
            [&](auto iR) {
                (void)iR;
                return StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>{};
            },
            Number<NumReduction>{});

        static_for<0, NumReduction, 1>{}([&](auto iR) {
            static_for<0, MThreadSliceSize, 1>{}(
                [&](auto J) { accu_value_buf_tuple(iR)(J) = identityVal; });
        });

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / block_group_size;
        const index_t block_local_id  = block_global_id % block_group_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        const index_t reduceSizePerBlock = K_BlockTileSize * num_k_block_tile_iteration;

        using ThreadBufferLengths         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<InDataType,
                                                                    AccDataType,
                                                                    InGridDesc_M_K,
                                                                    decltype(thread_buffer_desc),
                                                                    ThreadBufferLengths,
                                                                    ThreadBufferDimAccessOrder,
                                                                    InSrcVectorDim,
                                                                    InSrcVectorSize,
                                                                    1,
                                                                    false>(
            in_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        index_t reducedTiles = 0;
        do
        {
            threadwise_src_load.Run(in_grid_desc_m_k,
                                    in_global_val_buf,
                                    thread_buffer_desc,
                                    make_tuple(I0, I0),
                                    in_thread_buf);

            static_for<0, NumReduction, 1>{}([&](auto iR) {
                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    // do element-wise pre-reduction operation
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                        in_elementwise_op_tuple[iR](in_thread_buf_tuple(iR)(Number<offset>{}),
                                                    in_thread_buf(Number<offset>{}));
                    });
                });

                ThreadwiseReduce::Reduce(in_thread_buf_tuple(iR), accu_value_buf_tuple(iR));
            });

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            reducedTiles++;
        } while(reducedTiles < num_k_block_tile_iteration);

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        static_for<0, NumReduction, 1>{}([&](auto iR) {
            using OutDataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[iR])>;
            using OutDataType        = remove_cvref_t<remove_pointer_t<OutDataTypePointer>>;

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                BlockwiseReduce::Reduce(reduce_work_buf, accu_value_buf_tuple(iR)(I));
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if(thread_k_cluster_id == 0)
                {
                    acc_elementwise_op_tuple[iR](accu_value_buf_tuple(iR)(I),
                                                 accu_value_buf_tuple(iR)(I));

                    accu_value_buf_tuple(iR)(I) *= alpha_values[iR];
                }
            });

            if(thread_k_cluster_id == 0)
            {
                if(block_group_size == 0 && !float_equal_zero{}(beta_values[iR]))
                {
                    StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true>
                        priorDstValueBuf;

                    auto threadwise_dst_load =
                        ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                         OutDataType,
                                                         decltype(out_grid_desc_m_tuple[iR]),
                                                         decltype(reduced_data_desc),
                                                         Sequence<MThreadSliceSize>,
                                                         Sequence<0>,
                                                         0,
                                                         OutDstVectorSizeSeq::At(iR),
                                                         1,
                                                         false>(
                            out_grid_desc_m_tuple[iR],
                            make_multi_index(blkgroup_id * M_BlockTileSize +
                                             thread_m_cluster_id * MThreadSliceSize));

                    threadwise_dst_load.Run(out_grid_desc_m_tuple[iR],
                                            out_global_val_buf_tuple(iR),
                                            reduced_data_desc,
                                            make_tuple(I0),
                                            priorDstValueBuf);

                    static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                        accu_value_buf_tuple(iR)(I) +=
                            type_convert<AccDataType>(priorDstValueBuf[I]) * beta_values[iR];
                    });
                };

                auto threadwise_dst_store =
                    ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                       OutDataType,
                                                       decltype(reduced_data_desc),
                                                       decltype(out_grid_desc_m_tuple[iR]),
                                                       PassThroughOp,
                                                       Sequence<MThreadSliceSize>,
                                                       Sequence<0>,
                                                       0,
                                                       OutDstVectorSizeSeq::At(iR),
                                                       OutMemoryDataOperation,
                                                       1,
                                                       true>(
                        out_grid_desc_m_tuple[iR],
                        make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize),
                        PassThroughOp{});

                threadwise_dst_store.Run(reduced_data_desc,
                                         make_tuple(I0),
                                         accu_value_buf_tuple[iR],
                                         out_grid_desc_m_tuple[iR],
                                         out_global_val_buf_tuple(iR));
            };
        });
    };
}; // namespace ck

} // namespace ck
