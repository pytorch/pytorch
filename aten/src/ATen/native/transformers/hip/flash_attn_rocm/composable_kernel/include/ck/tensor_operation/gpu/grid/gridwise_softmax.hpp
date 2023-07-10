// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseReduction,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename GridDesc_M_K>
__global__ void kernel_softmax(const GridDesc_M_K in_grid_desc_m_k,
                               const GridDesc_M_K out_grid_desc_m_k,
                               index_t block_group_size,
                               index_t num_k_block_tile_iteration,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_in_value_global,
                               AccDataType beta,
                               OutDataType* const __restrict__ p_out_value_global)
{
    GridwiseReduction::Run(in_grid_desc_m_k,
                           out_grid_desc_m_k,
                           block_group_size,
                           num_k_block_tile_iteration,
                           alpha,
                           p_in_value_global,
                           beta,
                           p_out_value_global);
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize,
          bool SweepOnce>
struct GridwiseSoftmax_mk_to_mk
{
    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (KThreadSliceSize % OutDstVectorSize == 0),
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

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const GridDesc_M_K& in_grid_desc_m_k,
                               const GridDesc_M_K& out_grid_desc_m_k,
                               index_t block_group_size,
                               index_t num_k_block_tile_iteration,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_in_value_global,
                               AccDataType beta,
                               OutDataType* const __restrict__ p_out_value_global)
    {
        if constexpr(SweepOnce)
        {
            num_k_block_tile_iteration = 1;
        }

        // LDS
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        auto out_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m_k.GetElementSpaceSize());

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            out_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> max_value_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            max_value_buf(I) = reduce::Max::template GetIdentityValue<AccDataType>();
        });

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accu_value_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
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

        // Normally, 0 as invalid element value is adequate since 0 makes no contribution to
        // accumulated result. However, in stable softmax, all values 0s or not are subtracted by
        // another value_max. As numbers become non-zero, effectively it allows invalid values to
        // slip through and contribute to the accumulated result.
        //
        // The trick here is leveraging the fact that many math functions (add, sub, exp, ...)
        // propagate NaNs when operands have NaNs involved. By initialiing invalid element value
        // with NaN, an invalid value doing math manipulations is still NaN, which in turn can still
        // be identified as an invalid value. We can then discard the invalid values which
        // originally failed the bound check during accumulation. This allows to ignore values that
        // failed bound check even after multiple math manipulations.
        //
        // NOTE: reset coordinate after every step because the same threadwise copy will sweep
        // through global memory 3 times back and forth
        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<InDataType,
                                                                    AccDataType,
                                                                    GridDesc_M_K,
                                                                    decltype(thread_buffer_desc),
                                                                    ThreadBufferLengths,
                                                                    ThreadBufferDimAccessOrder,
                                                                    InSrcVectorDim,
                                                                    InSrcVectorSize,
                                                                    1,
                                                                    true /* ResetCoordAfterRun */,
                                                                    true /* InvalidElementAsNaN */>(
            in_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                                    AccDataType,
                                                                    GridDesc_M_K,
                                                                    decltype(thread_buffer_desc),
                                                                    ThreadBufferLengths,
                                                                    ThreadBufferDimAccessOrder,
                                                                    InSrcVectorDim,
                                                                    InSrcVectorSize,
                                                                    1,
                                                                    false>(
            out_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dst_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               OutDataType,
                                               decltype(thread_buffer_desc),
                                               GridDesc_M_K,
                                               PassThroughOp,
                                               ThreadBufferLengths,
                                               ThreadBufferDimAccessOrder,
                                               InSrcVectorDim,
                                               OutDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                out_grid_desc_m_k,
                make_multi_index(
                    blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                    block_local_id * reduceSizePerBlock + thread_k_cluster_id * KThreadSliceSize),
                PassThroughOp{});

        constexpr auto in_thread_copy_fwd_step =
            make_multi_index(0, SweepOnce ? 0 : K_BlockTileSize);
        constexpr auto in_thread_copy_bwd_step =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        ///
        /// max(x)
        ///
        using BlockwiseMaxReduce = PartitionedBlockwiseReduction<
            AccDataType,
            BlockSize,
            ThreadClusterLengths_M_K,
            ThreadClusterArrangeOrder,
            reduce::Max,
            false, // param ignored
            detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;

        using ThreadwiseMaxReduce =
            ThreadwiseReduction<AccDataType,
                                ThreadReduceSrcDesc_M_K,
                                ThreadReduceDstDesc_M,
                                reduce::Max,
                                false, // param ignored
                                detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global, in_grid_desc_m_k.GetElementSpaceSize());

        index_t reducedTiles = 0;
        do
        {
            threadwise_src_load.Run(in_grid_desc_m_k,
                                    in_global_val_buf,
                                    thread_buffer_desc,
                                    make_tuple(I0, I0),
                                    in_thread_buf);

            ThreadwiseMaxReduce::Reduce(in_thread_buf, max_value_buf);

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_fwd_step);

            reducedTiles++;
        } while(reducedTiles < num_k_block_tile_iteration);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            BlockwiseMaxReduce::Reduce(reduce_work_buf, max_value_buf(I));
            block_sync_lds();
        });

        threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_bwd_step);

        ///
        /// sum(exp(x - max(x)))
        ///
        using BlockwiseSumReduce = PartitionedBlockwiseReduction<
            AccDataType,
            BlockSize,
            ThreadClusterLengths_M_K,
            ThreadClusterArrangeOrder,
            reduce::Add,
            false, // ignored
            detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;

        using ThreadwiseSumReduce =
            ThreadwiseReduction<AccDataType,
                                ThreadReduceSrcDesc_M_K,
                                ThreadReduceDstDesc_M,
                                reduce::Add,
                                false, // ignored
                                detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;

        reducedTiles = 0;
        do
        {
            if constexpr(!SweepOnce)
            {
                threadwise_src_load.Run(in_grid_desc_m_k,
                                        in_global_val_buf,
                                        thread_buffer_desc,
                                        make_tuple(I0, I0),
                                        in_thread_buf);
            }

            // do element-wise pre-reduction operation
            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset = thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                    out_thread_buf(Number<offset>{}) =
                        math::exp(in_thread_buf(Number<offset>{}) - max_value_buf(iM));
                });
            });

            ThreadwiseSumReduce::Reduce(out_thread_buf, accu_value_buf);

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_bwd_step);

            reducedTiles++;
        } while(reducedTiles < num_k_block_tile_iteration);

        block_sync_lds(); // wait for reading being complete before writing to LDS
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            BlockwiseSumReduce::Reduce(reduce_work_buf, accu_value_buf(I));
            block_sync_lds();
        });

        threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_fwd_step);

        ///
        /// softmax
        ///
        reducedTiles = 0;
        if(float_equal_zero{}(beta))
        {
            do
            {
                if constexpr(!SweepOnce)
                {
                    threadwise_src_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_buf);
                }

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    // out = alpha * exp(x - max(x)) / sum(exp(x - max(x)))
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                        out_thread_buf(Number<offset>{}) =
                            alpha * math::exp(in_thread_buf(Number<offset>{}) - max_value_buf(iM)) /
                            accu_value_buf(iM);
                    });
                });

                threadwise_dst_store.Run(thread_buffer_desc,
                                         make_tuple(I0, I0),
                                         out_thread_buf,
                                         out_grid_desc_m_k,
                                         out_global_val_buf);

                threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_fwd_step);
                threadwise_dst_store.MoveDstSliceWindow(out_grid_desc_m_k, in_thread_copy_fwd_step);

                reducedTiles++;
            } while(reducedTiles < num_k_block_tile_iteration);
        }
        else
        {
            StaticBuffer<AddressSpaceEnum::Vgpr,
                         AccDataType,
                         MThreadSliceSize * KThreadSliceSize,
                         true>
                in_prior_dst_buf;
            do
            {
                if constexpr(!SweepOnce)
                {
                    threadwise_src_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_buf);
                }
                threadwise_dst_load.Run(out_grid_desc_m_k,
                                        out_global_val_buf,
                                        thread_buffer_desc,
                                        make_tuple(I0, I0),
                                        in_prior_dst_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    // out = alpha * exp(x - max(x)) / sum(exp(x - max(x))) + beta * prior_out
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                        out_thread_buf(Number<offset>{}) =
                            alpha * math::exp(in_thread_buf(Number<offset>{}) - max_value_buf(iM)) /
                                accu_value_buf(iM) +
                            beta * in_prior_dst_buf(Number<offset>{});
                    });
                });

                threadwise_dst_store.Run(thread_buffer_desc,
                                         make_tuple(I0, I0),
                                         out_thread_buf,
                                         out_grid_desc_m_k,
                                         out_global_val_buf);

                threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_fwd_step);
                threadwise_dst_store.MoveDstSliceWindow(out_grid_desc_m_k, in_thread_copy_fwd_step);
                threadwise_dst_load.MoveSrcSliceWindow(out_grid_desc_m_k, in_thread_copy_fwd_step);

                reducedTiles++;
            } while(reducedTiles < num_k_block_tile_iteration);
        }
    }
};

} // namespace ck
