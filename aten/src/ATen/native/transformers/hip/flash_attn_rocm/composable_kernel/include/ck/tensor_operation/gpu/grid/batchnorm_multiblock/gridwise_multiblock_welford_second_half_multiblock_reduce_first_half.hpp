// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseWelfordSecondHalfReduceFirstHalf_,
          typename XDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          typename XYGridDesc_M_K,
          typename MeanVarGridDesc_M,
          typename MeanVarCountGridDesc_M_K,
          typename DscaleDbiasGridDesc_M_G>
__global__ void kernel_welford_second_half_reduce_first_half(
    const XYGridDesc_M_K x_grid_desc_m_k,
    const XYGridDesc_M_K dy_grid_desc_m_k,
    const MeanVarGridDesc_M mean_var_grid_desc_m,
    const MeanVarCountGridDesc_M_K mean_var_count_grid_desc_m_k,
    const DscaleDbiasGridDesc_M_G dscale_dbias_grid_desc_m_g,
    index_t blkgroup_size,
    index_t num_xy_k_block_tile_iteration,
    index_t num_mean_var_count_k_block_tile_iteration,
    AccDataType epsilon,
    bool haveSavedMeanInvVar,
    const MeanVarDataType* const __restrict__ p_savedMean,
    const MeanVarDataType* const __restrict__ p_savedInvVar,
    const MeanVarDataType* const __restrict__ p_in_welford_mean,
    const MeanVarDataType* const __restrict__ p_in_welford_variance,
    const int32_t* const __restrict__ p_in_welford_count,
    const DyElementwiseOp dy_elementwise_op,
    MeanVarDataType* const __restrict__ p_out_welford_mean,
    MeanVarDataType* const __restrict__ p_out_welford_inv_variance,
    const XDataType* const __restrict__ p_x,
    const DyDataType* const __restrict__ p_dy,
    DscaleDbiasDataType* const __restrict__ p_reduce_dscale,
    DscaleDbiasDataType* const __restrict__ p_reduce_dbias)
{
    GridwiseWelfordSecondHalfReduceFirstHalf_::Run(x_grid_desc_m_k,
                                                   dy_grid_desc_m_k,
                                                   mean_var_grid_desc_m,
                                                   mean_var_count_grid_desc_m_k,
                                                   dscale_dbias_grid_desc_m_g,
                                                   blkgroup_size,
                                                   num_xy_k_block_tile_iteration,
                                                   num_mean_var_count_k_block_tile_iteration,
                                                   epsilon,
                                                   haveSavedMeanInvVar,
                                                   p_savedMean,
                                                   p_savedInvVar,
                                                   p_in_welford_mean,
                                                   p_in_welford_variance,
                                                   p_in_welford_count,
                                                   dy_elementwise_op,
                                                   p_out_welford_mean,
                                                   p_out_welford_inv_variance,
                                                   p_x,
                                                   p_dy,
                                                   p_reduce_dscale,
                                                   p_reduce_dbias);
};

template <typename XDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          typename XYGridDesc_M_K,
          typename MeanVarGridDesc_M,
          typename MeanVarCountGridDesc_M_K,
          typename DscaleDbiasGridDesc_M_G,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XDyVectorDim,
          index_t XSrcVectorSize,
          index_t DySrcVectorSize,
          index_t MeanVarSrcVectorSize>
struct GridwiseWelfordSecondHalfReduceFirstHalf
{
    static_assert((XDyVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0 &&
                   MThreadSliceSize % DySrcVectorSize == 0) ||
                      (XDyVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0 &&
                       KThreadSliceSize % DySrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XDyVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceSrcDesc_M_1 = decltype(
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}, Number<1>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelfordMerge<AccDataType, ThreadReduceSrcDesc_M_1, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<AccDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_K,
                                              ThreadClusterArrangeOrder>;

    using BlockwiseReduce = PartitionedBlockwiseReduction<AccDataType,
                                                          BlockSize,
                                                          ThreadClusterLengths_M_K,
                                                          ThreadClusterArrangeOrder,
                                                          ck::reduce::Add,
                                                          false>;

    using ThreadwiseReduce = ThreadwiseReduction<AccDataType,
                                                 ThreadReduceSrcDesc_M_K,
                                                 ThreadReduceDstDesc_M,
                                                 ck::reduce::Add,
                                                 false>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    // clang-format off
    // Two of the steps of Multiblock BatchNorm Backward
    // Step 1: Second half of Welford method to calculate mean and variance, as well as getting inv-variance = 1/sqrt(epsilon+variance) 
    // Step 2: First half of Reduction: dbias = sum(dy), dscale = sum(dy * (x-mean) * inv-variance)
    // clang-format on
    __device__ static void Run(const XYGridDesc_M_K& x_grid_desc_m_k,
                               const XYGridDesc_M_K& dy_grid_desc_m_k,
                               const MeanVarGridDesc_M& mean_var_grid_desc_m,
                               const MeanVarCountGridDesc_M_K& mean_var_count_grid_desc_m_k,
                               const DscaleDbiasGridDesc_M_G& dscale_dbias_grid_desc_m_g,
                               index_t blkgroup_size,
                               index_t num_xy_k_block_tile_iteration,
                               index_t num_mean_var_count_k_block_tile_iteration,
                               AccDataType epsilon,
                               bool haveSavedMeanInvVar,
                               const MeanVarDataType* const __restrict__ p_savedMean,
                               const MeanVarDataType* const __restrict__ p_savedInvVar,
                               const MeanVarDataType* const __restrict__ p_in_welford_mean,
                               const MeanVarDataType* const __restrict__ p_in_welford_variance,
                               const int32_t* const __restrict__ p_in_welford_count,
                               const DyElementwiseOp dy_elementwise_op,
                               MeanVarDataType* const __restrict__ p_out_welford_mean,
                               MeanVarDataType* const __restrict__ p_out_welford_inv_variance,
                               const XDataType* const __restrict__ p_x,
                               const DyDataType* const __restrict__ p_dy,
                               DscaleDbiasDataType* const __restrict__ p_reduce_dscale,
                               DscaleDbiasDataType* const __restrict__ p_reduce_dbias)
    {
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * 1, true>
            in_welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * 1, true>
            in_welford_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize * 1, true>
            in_welford_count_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            welford_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            welford_count_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>& mean_thread_buf =
            welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>&
            inv_var_thread_buf = welford_var_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            dy_thread_buf;

        // buffer of values of dy * (x-mean) * inv-variance, used as input of Blockwise reduction
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            tmp1_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            reduce_dscale_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            reduce_dbias_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / blkgroup_size;
        const index_t block_local_id  = block_global_id % blkgroup_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        using ThreadBufferLengths_M           = Sequence<MThreadSliceSize>;
        using ThreadBufferLengths_M_1         = Sequence<MThreadSliceSize, 1>;
        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m_1 = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

        // clang-format off
        // Step 1: load existing mean and inv-variance, or do final welford reduction on mean and variance as well as get inv-variance = 1/sqrt(epsilon+variance)
        // clang-format on

        if(haveSavedMeanInvVar)
        {
            const auto mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_savedMean, mean_var_grid_desc_m.GetElementSpaceSize());

            const auto inv_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_savedInvVar, mean_var_grid_desc_m.GetElementSpaceSize());

            auto threadwise_mean_inv_var_load =
                ThreadwiseTensorSliceTransfer_v2<MeanVarDataType,
                                                 AccDataType,
                                                 MeanVarGridDesc_M,
                                                 decltype(thread_buffer_desc_m),
                                                 ThreadBufferLengths_M,
                                                 Sequence<0>,
                                                 0,
                                                 MeanVarSrcVectorSize,
                                                 1,
                                                 true>(
                    mean_var_grid_desc_m,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize));

            threadwise_mean_inv_var_load.Run(mean_var_grid_desc_m,
                                             mean_global_buf,
                                             thread_buffer_desc_m,
                                             make_tuple(I0),
                                             mean_thread_buf);

            threadwise_mean_inv_var_load.Run(mean_var_grid_desc_m,
                                             inv_var_global_buf,
                                             thread_buffer_desc_m,
                                             make_tuple(I0),
                                             inv_var_thread_buf);
        }
        else
        {
            const auto welford_mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_in_welford_mean, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

            const auto welford_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_in_welford_variance, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

            const auto welford_count_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_in_welford_count, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

            auto threadwise_mean_var_load_m_k =
                ThreadwiseTensorSliceTransfer_v2<AccDataType,
                                                 AccDataType,
                                                 MeanVarCountGridDesc_M_K,
                                                 decltype(thread_buffer_desc_m_1),
                                                 ThreadBufferLengths_M_1,
                                                 Sequence<0, 1>,
                                                 1,
                                                 1,
                                                 1,
                                                 true>(
                    mean_var_count_grid_desc_m_k,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     thread_k_cluster_id * 1));

            auto threadwise_count_load_m_k =
                ThreadwiseTensorSliceTransfer_v2<int32_t,
                                                 int32_t,
                                                 MeanVarCountGridDesc_M_K,
                                                 decltype(thread_buffer_desc_m_1),
                                                 ThreadBufferLengths_M_1,
                                                 Sequence<0, 1>,
                                                 1,
                                                 1,
                                                 1,
                                                 true>(
                    mean_var_count_grid_desc_m_k,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     thread_k_cluster_id * 1));

            constexpr auto mean_var_count_thread_copy_step_m_k =
                make_multi_index(0, KThreadClusterSize * 1);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                welford_mean_thread_buf(I)  = type_convert<AccDataType>(0.0f);
                welford_var_thread_buf(I)   = type_convert<AccDataType>(0.0f);
                welford_count_thread_buf(I) = 0;
            });

            for(index_t reducedTiles = 0; reducedTiles < num_mean_var_count_k_block_tile_iteration;
                ++reducedTiles)
            {
                threadwise_mean_var_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                                 welford_mean_global_buf,
                                                 thread_buffer_desc_m_1,
                                                 make_tuple(I0, I0),
                                                 in_welford_mean_thread_buf);

                threadwise_mean_var_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                                 welford_var_global_buf,
                                                 thread_buffer_desc_m_1,
                                                 make_tuple(I0, I0),
                                                 in_welford_var_thread_buf);

                threadwise_count_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                              welford_count_global_buf,
                                              thread_buffer_desc_m_1,
                                              make_tuple(I0, I0),
                                              in_welford_count_thread_buf);

                ThreadwiseWelford::Run(in_welford_mean_thread_buf,
                                       in_welford_var_thread_buf,
                                       in_welford_count_thread_buf,
                                       welford_mean_thread_buf,
                                       welford_var_thread_buf,
                                       welford_count_thread_buf);

                threadwise_mean_var_load_m_k.MoveSrcSliceWindow(
                    mean_var_count_grid_desc_m_k, mean_var_count_thread_copy_step_m_k);
                threadwise_count_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_k,
                                                             mean_var_count_thread_copy_step_m_k);
            }

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseWelford::Run(welford_mean_thread_buf(I),
                                      welford_var_thread_buf(I),
                                      welford_count_thread_buf(I));
            });

            // calculate inv-variance as 1/sqrt(epsilon+variance), stored in place of variance
            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                welford_var_thread_buf(I) =
                    type_convert<AccDataType>(1.0) / sqrt(welford_var_thread_buf[I] + epsilon);
            });

            if(block_local_id == 0 && thread_k_cluster_id == 0)
            {

                auto threadwise_mean_inv_var_store =
                    ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                       MeanVarDataType,
                                                       decltype(thread_buffer_desc_m),
                                                       MeanVarGridDesc_M,
                                                       PassThroughOp,
                                                       ThreadBufferLengths_M,
                                                       Sequence<0>,
                                                       0,
                                                       1,
                                                       InMemoryDataOperationEnum::Set,
                                                       1,
                                                       true>(
                        mean_var_grid_desc_m,
                        make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize),
                        PassThroughOp{});

                auto mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_welford_mean, mean_var_grid_desc_m.GetElementSpaceSize());

                auto inv_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_welford_inv_variance, mean_var_grid_desc_m.GetElementSpaceSize());

                threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                                  make_tuple(I0),
                                                  mean_thread_buf,
                                                  mean_var_grid_desc_m,
                                                  mean_global_buf);

                threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                                  make_tuple(I0),
                                                  inv_var_thread_buf,
                                                  mean_var_grid_desc_m,
                                                  inv_var_global_buf);
            };
        };

        const index_t workSizePerBlock = K_BlockTileSize * num_xy_k_block_tile_iteration;

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  XYGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XDyVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             workSizePerBlock * block_local_id +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dy_load = ThreadwiseTensorSliceTransfer_v2<DyDataType,
                                                                   AccDataType,
                                                                   XYGridDesc_M_K,
                                                                   decltype(thread_buffer_desc_m_k),
                                                                   ThreadBufferLengths_M_K,
                                                                   ThreadBufferDimAccessOrder,
                                                                   XDyVectorDim,
                                                                   DySrcVectorSize,
                                                                   1,
                                                                   true>(
            dy_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             workSizePerBlock * block_local_id +
                                 thread_k_cluster_id * KThreadSliceSize));

        const auto x_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x, x_grid_desc_m_k.GetElementSpaceSize());

        const auto dy_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dy, dy_grid_desc_m_k.GetElementSpaceSize());

        constexpr auto xy_thread_copy_step_m_k = make_multi_index(0, K_BlockTileSize);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            reduce_dscale_thread_buf(I) = type_convert<AccDataType>(0);
            reduce_dbias_thread_buf(I)  = type_convert<AccDataType>(0);
        });

        // clang-format off
        // Step 2: first-half of reduction: dbias = sum(dy), dscale = sum(dy * (x-mean) * inv-variance)
        // clang-format on

        for(index_t reducedTiles = 0; reducedTiles < num_xy_k_block_tile_iteration; ++reducedTiles)
        {
            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            threadwise_dy_load.Run(dy_grid_desc_m_k,
                                   dy_global_buf,
                                   thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   dy_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    dy_elementwise_op(dy_thread_buf(Number<offset>{}),
                                      dy_thread_buf[Number<offset>{}]);

                    AccDataType norm_x = (x_thread_buf[Number<offset>{}] - mean_thread_buf[iM]) *
                                         inv_var_thread_buf[iM];

                    tmp1_thread_buf(Number<offset>{}) = norm_x * dy_thread_buf[Number<offset>{}];
                });
            });

            ThreadwiseReduce::Reduce(tmp1_thread_buf, reduce_dscale_thread_buf);
            ThreadwiseReduce::Reduce(dy_thread_buf, reduce_dbias_thread_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, xy_thread_copy_step_m_k);
            threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, xy_thread_copy_step_m_k);
        };

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseReduce::Reduce(reduce_work_buf, reduce_dscale_thread_buf(I));
            block_sync_lds();
            BlockwiseReduce::Reduce(reduce_work_buf, reduce_dbias_thread_buf(I));
        });

        auto threadwise_dscale_dbias_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               DscaleDbiasDataType,
                                               decltype(thread_buffer_desc_m_1),
                                               DscaleDbiasGridDesc_M_G,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_1,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                dscale_dbias_grid_desc_m_g,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_local_id),
                PassThroughOp{});

        auto reduce_dscale_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_reduce_dscale, dscale_dbias_grid_desc_m_g.GetElementSpaceSize());

        auto reduce_dbias_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_reduce_dbias, dscale_dbias_grid_desc_m_g.GetElementSpaceSize());

        if(thread_k_cluster_id == 0)
        {
            threadwise_dscale_dbias_store.Run(thread_buffer_desc_m_1,
                                              make_tuple(I0, I0),
                                              reduce_dscale_thread_buf,
                                              dscale_dbias_grid_desc_m_g,
                                              reduce_dscale_global_buf);

            threadwise_dscale_dbias_store.Run(thread_buffer_desc_m_1,
                                              make_tuple(I0, I0),
                                              reduce_dbias_thread_buf,
                                              dscale_dbias_grid_desc_m_g,
                                              reduce_dbias_global_buf);
        };
    };
};

} // namespace ck
