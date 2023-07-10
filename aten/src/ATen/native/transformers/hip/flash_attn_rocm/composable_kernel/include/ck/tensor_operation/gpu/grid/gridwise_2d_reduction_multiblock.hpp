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

template <typename GridwiseReduction,
          bool OutputIndex,
          bool HaveIndexInput,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void kernel_reduce_multiblock(const InGridDesc_M_K in_grid_desc_m_k,
                                         const OutGridDesc_M out_grid_desc_m,
                                         const InElementwiseOperation in_elementwise_op,
                                         const AccElementwiseOperation acc_elementwise_op,
                                         index_t block_group_size,
                                         index_t num_k_block_tile_iteration,
                                         AccDataType alpha,
                                         const InDataType* const __restrict__ p_in_value_global,
                                         const IndexDataType* const __restrict__ p_in_index_global,
                                         AccDataType beta,
                                         OutDataType* const __restrict__ p_out_value_global,
                                         IndexDataType* const __restrict__ p_out_index_global)
{
    if constexpr(!OutputIndex)
    {
        (void)p_in_index_global;
        (void)p_out_index_global;

        GridwiseReduction::Run(in_grid_desc_m_k,
                               out_grid_desc_m,
                               in_elementwise_op,
                               acc_elementwise_op,
                               block_group_size,
                               num_k_block_tile_iteration,
                               alpha,
                               p_in_value_global,
                               beta,
                               p_out_value_global);
    }
    else
    {
        GridwiseReduction::template RunWithIndex<HaveIndexInput>(in_grid_desc_m_k,
                                                                 out_grid_desc_m,
                                                                 in_elementwise_op,
                                                                 acc_elementwise_op,
                                                                 num_k_block_tile_iteration,
                                                                 alpha,
                                                                 p_in_value_global,
                                                                 p_in_index_global,
                                                                 beta,
                                                                 p_out_value_global,
                                                                 p_out_index_global);
    };
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct GridwiseReduction_mk_to_m_multiblock
{
    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

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
                               const OutGridDesc_M& out_grid_desc_m,
                               const InElementwiseOperation& in_elementwise_op,
                               const AccElementwiseOperation& acc_elementwise_op,
                               index_t block_group_size,
                               index_t num_k_block_tile_iteration,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_in_value_global,
                               AccDataType beta,
                               OutDataType* const __restrict__ p_out_value_global)
    {
        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        // LDS
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        auto out_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m.GetElementSpaceSize());

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accu_value_buf(I) = identityVal; });

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

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                // do element-wise pre-reduction operation
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset = thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                    in_elementwise_op(in_thread_buf(Number<offset>{}),
                                      in_thread_buf(Number<offset>{}));
                });
            });

            ThreadwiseReduce::Reduce(in_thread_buf, accu_value_buf);

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            reducedTiles++;
        } while(reducedTiles < num_k_block_tile_iteration);

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        static_for<0, MThreadSliceSize, 1>{}(
            [&](auto I) { BlockwiseReduce::Reduce(reduce_work_buf, accu_value_buf(I)); });

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if(thread_k_cluster_id == 0)
            {
                acc_elementwise_op(accu_value_buf(I), accu_value_buf(I));

                accu_value_buf(I) *= alpha;
            }
        });

        if(thread_k_cluster_id == 0)
        {
            if(block_group_size == 0 && !float_equal_zero{}(beta))
            {
                StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValueBuf;

                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                     OutDataType,
                                                     OutGridDesc_M,
                                                     decltype(reduced_data_desc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     OutDstVectorSize,
                                                     1,
                                                     false>(
                        out_grid_desc_m,
                        make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize));

                threadwise_dst_load.Run(out_grid_desc_m,
                                        out_global_val_buf,
                                        reduced_data_desc,
                                        make_tuple(I0),
                                        priorDstValueBuf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                    accu_value_buf(I) += type_convert<AccDataType>(priorDstValueBuf[I]) * beta;
                });
            };

            auto threadwise_dst_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   OutDataType,
                                                   decltype(reduced_data_desc),
                                                   OutGridDesc_M,
                                                   PassThroughOp,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutDstVectorSize,
                                                   OutMemoryDataOperation,
                                                   1,
                                                   true>(
                    out_grid_desc_m,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            threadwise_dst_store.Run(reduced_data_desc,
                                     make_tuple(I0),
                                     accu_value_buf,
                                     out_grid_desc_m,
                                     out_global_val_buf);
        }
    };

    template <bool HaveIndexInput>
    __device__ static void RunWithIndex(const InGridDesc_M_K& in_grid_desc_m_k,
                                        const OutGridDesc_M& out_grid_desc_m,
                                        const InElementwiseOperation in_elementwise_op,
                                        const AccElementwiseOperation acc_elementwise_op,
                                        index_t num_k_block_tile_iteration,
                                        AccDataType alpha,
                                        const InDataType* const __restrict__ p_in_value_global,
                                        const IndexDataType* const __restrict__ p_in_index_global,
                                        AccDataType beta,
                                        OutDataType* const __restrict__ p_out_value_global,
                                        IndexDataType* const __restrict__ p_out_index_global)
    {
        using BlockwiseReduceWithIndex =
            PartitionedBlockwiseReductionWithIndex<AccDataType,
                                                   IndexDataType,
                                                   BlockSize,
                                                   Sequence<MThreadClusterSize, KThreadClusterSize>,
                                                   ThreadClusterArrangeOrder,
                                                   ReduceOperation,
                                                   PropagateNan>;

        using AccumulationWithIndex = detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                             ReduceOperation,
                                                                             AccDataType,
                                                                             IndexDataType>;

        (void)in_elementwise_op;

        // LDS
        __shared__ AccDataType p_reduce_work_val_buffer[BlockSize];
        __shared__ IndexDataType p_reduce_work_idx_buffer[BlockSize];

        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        const auto in_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_index_global, in_grid_desc_m_k.GetElementSpaceSize());
        auto out_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m.GetElementSpaceSize());
        auto out_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_index_global, out_grid_desc_m.GetElementSpaceSize());

        auto reduce_work_val_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_val_buffer, BlockSize);
        auto reduce_work_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     IndexDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, IndexDataType, MThreadSliceSize, true> accu_index_buf;

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_val_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
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
                make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accu_value_buf(I) = identityVal;
            accu_index_buf(I) = 0;
        });

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        index_t reducedTiles = 0;

        if constexpr(HaveIndexInput)
        {
            auto threadwise_src_idx_load =
                ThreadwiseTensorSliceTransfer_v2<IndexDataType,
                                                 IndexDataType,
                                                 InGridDesc_M_K,
                                                 decltype(thread_buffer_desc),
                                                 ThreadBufferLengths,
                                                 ThreadBufferDimAccessOrder,
                                                 InSrcVectorDim,
                                                 InSrcVectorSize,
                                                 1,
                                                 false>(
                    in_grid_desc_m_k,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     thread_k_cluster_id * KThreadSliceSize));

            do
            {
                // load the thread slice
                threadwise_src_val_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_val_buf);
                threadwise_src_idx_load.Run(in_grid_desc_m_k,
                                            in_global_idx_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_idx_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    AccDataType tmpValue   = identityVal;
                    IndexDataType tmpIndex = 0;

                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));

                        AccumulationWithIndex::Calculate(tmpValue,
                                                         in_thread_val_buf[Number<offset>{}],
                                                         tmpIndex,
                                                         in_thread_idx_buf[Number<offset>{}]);
                    });

                    BlockwiseReduceWithIndex::Reduce(
                        reduce_work_val_buf, reduce_work_idx_buf, tmpValue, tmpIndex);

                    AccumulationWithIndex::Calculate(
                        accu_value_buf(iM), tmpValue, accu_index_buf(iM), tmpIndex);
                });

                threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);
                threadwise_src_idx_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

                reducedTiles++;
            } while(reducedTiles < num_k_block_tile_iteration);
        }
        else
        {
            index_t indexOffset = 0;

            do
            {
                // load the thread slice
                threadwise_src_val_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_val_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));

                        // initialize the indices for the per-thread to-reduce values
                        in_thread_idx_buf(Number<offset>{}) =
                            indexOffset + thread_k_cluster_id * KThreadSliceSize + iK();

                        // do element-wise pre-reduction operation
                        in_elementwise_op(in_thread_val_buf(Number<offset>{}),
                                          in_thread_val_buf(Number<offset>{}));
                    });

                    AccDataType tmpValue   = identityVal;
                    IndexDataType tmpIndex = 0;

                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));

                        AccumulationWithIndex::Calculate(tmpValue,
                                                         in_thread_val_buf[Number<offset>{}],
                                                         tmpIndex,
                                                         in_thread_idx_buf[Number<offset>{}]);
                    });

                    BlockwiseReduceWithIndex::Reduce(
                        reduce_work_val_buf, reduce_work_idx_buf, tmpValue, tmpIndex);

                    AccumulationWithIndex::Calculate(
                        accu_value_buf(iM), tmpValue, accu_index_buf(iM), tmpIndex);
                });

                threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

                indexOffset += K_BlockTileSize;
                reducedTiles++;
            } while(reducedTiles < num_k_block_tile_iteration);
        };

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if(thread_k_cluster_id == 0)
            {
                // for indiced operation, acc_elementwise_op shoud do nothing
                acc_elementwise_op(accu_value_buf(I), accu_value_buf(I));

                accu_value_buf(I) *= alpha;
            }
        });

        if(thread_k_cluster_id == 0)
        {
            if(!float_equal_zero{}(beta))
            {
                StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValueBuf;

                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                     OutDataType,
                                                     OutGridDesc_M,
                                                     decltype(reduced_data_desc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     OutDstVectorSize,
                                                     1,
                                                     true>(
                        out_grid_desc_m,
                        make_multi_index(block_global_1d_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize));

                threadwise_dst_load.Run(out_grid_desc_m,
                                        out_global_val_buf,
                                        reduced_data_desc,
                                        make_tuple(I0),
                                        priorDstValueBuf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                    accu_value_buf(I) += type_convert<AccDataType>(priorDstValueBuf[I]) * beta;
                });
            };

            auto threadwise_dst_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   OutDataType,
                                                   decltype(reduced_data_desc),
                                                   OutGridDesc_M,
                                                   PassThroughOp,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutDstVectorSize,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>(
                    out_grid_desc_m,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            auto threadwise_dst_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<IndexDataType,
                                                   IndexDataType,
                                                   decltype(reduced_data_desc),
                                                   OutGridDesc_M,
                                                   PassThroughOp,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutDstVectorSize,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>(
                    out_grid_desc_m,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            threadwise_dst_val_store.Run(reduced_data_desc,
                                         make_tuple(I0),
                                         accu_value_buf,
                                         out_grid_desc_m,
                                         out_global_val_buf);
            threadwise_dst_idx_store.Run(reduced_data_desc,
                                         make_tuple(I0),
                                         accu_index_buf,
                                         out_grid_desc_m,
                                         out_global_idx_buf);
        }
    };
};

} // namespace ck
