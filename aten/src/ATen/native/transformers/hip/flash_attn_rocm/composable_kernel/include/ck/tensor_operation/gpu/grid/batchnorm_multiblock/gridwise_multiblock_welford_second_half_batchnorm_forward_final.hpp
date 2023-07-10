// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseWelfordSecondHalfBatchNormForwardFinal_,
          typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          typename XYGridDesc_M_K,
          typename MeanVarCountGridDesc_M_K,
          typename ScaleBiasGridDesc_M,
          typename MeanVarGridDesc_M>
__global__ void kernel_welford_second_half_batchnorm_forward_final(
    const XYGridDesc_M_K x_grid_desc_m_k,
    const XYGridDesc_M_K y_grid_desc_m_k,
    const MeanVarCountGridDesc_M_K mean_var_count_grid_desc_m_k,
    const ScaleBiasGridDesc_M scale_grid_desc_m,
    const ScaleBiasGridDesc_M bias_grid_desc_m,
    const MeanVarGridDesc_M mean_var_grid_desc_m,
    index_t blkgroup_size,
    index_t num_xy_k_block_tile_iteration,
    index_t num_mean_var_count_k_block_tile_iteration,
    AccDataType epsilon,
    const MeanVarDataType* const __restrict__ p_in_welford_mean,
    const MeanVarDataType* const __restrict__ p_in_welford_variance,
    const int32_t* const __restrict__ p_in_welford_count,
    const XDataType* const __restrict__ p_x,
    const ScaleDataType* const __restrict__ p_scale,
    const BiasDataType* const __restrict__ p_bias,
    const YElementwiseOp y_elementwise_op,
    YDataType* const __restrict__ p_y,
    bool updateMovingAverage,
    AccDataType averageFactor,
    MeanVarDataType* const __restrict__ resultRunningMean,
    MeanVarDataType* const __restrict__ resultRunningVariance,
    bool saveMeanInvVariance,
    MeanVarDataType* const __restrict__ resultSaveMean,
    MeanVarDataType* const __restrict__ resultSaveInvVariance)
{
    GridwiseWelfordSecondHalfBatchNormForwardFinal_::Run(x_grid_desc_m_k,
                                                         y_grid_desc_m_k,
                                                         mean_var_count_grid_desc_m_k,
                                                         scale_grid_desc_m,
                                                         bias_grid_desc_m,
                                                         mean_var_grid_desc_m,
                                                         blkgroup_size,
                                                         num_xy_k_block_tile_iteration,
                                                         num_mean_var_count_k_block_tile_iteration,
                                                         epsilon,
                                                         p_in_welford_mean,
                                                         p_in_welford_variance,
                                                         p_in_welford_count,
                                                         p_x,
                                                         p_scale,
                                                         p_bias,
                                                         y_elementwise_op,
                                                         p_y,
                                                         updateMovingAverage,
                                                         averageFactor,
                                                         resultRunningMean,
                                                         resultRunningVariance,
                                                         saveMeanInvVariance,
                                                         resultSaveMean,
                                                         resultSaveInvVariance);
};

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          typename XYGridDesc_M_K,
          typename MeanVarCountGridDesc_M_K,
          typename ScaleBiasGridDesc_M,
          typename MeanVarGridDesc_M,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcYDstVectorDim,
          index_t XSrcVectorSize,
          index_t YDstVectorSize,
          index_t ScaleSrcVectorSize,
          index_t BiasSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct GridwiseWelfordSecondHalfBatchNormForwardFinal
{
    static_assert((XSrcYDstVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcYDstVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((XSrcYDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (XSrcYDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XSrcYDstVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

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

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const XYGridDesc_M_K& x_grid_desc_m_k,
                               const XYGridDesc_M_K& y_grid_desc_m_k,
                               const MeanVarCountGridDesc_M_K& mean_var_count_grid_desc_m_k,
                               const ScaleBiasGridDesc_M& scale_grid_desc_m,
                               const ScaleBiasGridDesc_M& bias_grid_desc_m,
                               const MeanVarGridDesc_M& mean_var_grid_desc_m,
                               index_t blkgroup_size,
                               index_t num_xy_k_block_tile_iteration,
                               index_t num_mean_var_count_k_block_tile_iteration,
                               AccDataType epsilon,
                               const MeanVarDataType* const __restrict__ p_in_welford_mean,
                               const MeanVarDataType* const __restrict__ p_in_welford_variance,
                               const int32_t* const __restrict__ p_in_welford_count,
                               const XDataType* const __restrict__ p_x,
                               const ScaleDataType* const __restrict__ p_scale,
                               const BiasDataType* const __restrict__ p_bias,
                               const YElementwiseOp y_elementwise_op,
                               YDataType* const __restrict__ p_y,
                               bool updateMovingAverage,
                               AccDataType averageFactor,
                               MeanVarDataType* const __restrict__ resultRunningMean,
                               MeanVarDataType* const __restrict__ resultRunningVariance,
                               bool saveMeanInvVariance,
                               MeanVarDataType* const __restrict__ resultSaveMean,
                               MeanVarDataType* const __restrict__ resultSaveInvVariance)

    {
        using ck::math::sqrt;

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

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> scale_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> bias_thread_buf;

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

        auto threadwise_mean_var_load_m_k =
            ThreadwiseTensorSliceTransfer_v2<MeanVarDataType,
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

        const auto welford_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_mean, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

        const auto welford_var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_variance, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

        const auto welford_count_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_count, mean_var_count_grid_desc_m_k.GetElementSpaceSize());

        constexpr auto mean_var_count_thread_copy_step_m_k =
            make_multi_index(0, KThreadClusterSize * 1);

        // Step 1: do final welford reduction to get mean and variance

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            welford_mean_thread_buf(I)  = type_convert<AccDataType>(0.0f);
            welford_var_thread_buf(I)   = type_convert<AccDataType>(0.0f);
            welford_count_thread_buf(I) = 0;
        });

        for(index_t reducedTiles = 0; reducedTiles < num_mean_var_count_k_block_tile_iteration;
            ++reducedTiles)
        {
            threadwise_mean_var_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                             welford_mean_global_val_buf,
                                             thread_buffer_desc_m_1,
                                             make_tuple(I0, I0),
                                             in_welford_mean_thread_buf);

            threadwise_mean_var_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                             welford_var_global_val_buf,
                                             thread_buffer_desc_m_1,
                                             make_tuple(I0, I0),
                                             in_welford_var_thread_buf);

            threadwise_count_load_m_k.Run(mean_var_count_grid_desc_m_k,
                                          welford_count_global_val_buf,
                                          thread_buffer_desc_m_1,
                                          make_tuple(I0, I0),
                                          in_welford_count_thread_buf);

            ThreadwiseWelford::Run(in_welford_mean_thread_buf,
                                   in_welford_var_thread_buf,
                                   in_welford_count_thread_buf,
                                   welford_mean_thread_buf,
                                   welford_var_thread_buf,
                                   welford_count_thread_buf);

            threadwise_mean_var_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_k,
                                                            mean_var_count_thread_copy_step_m_k);
            threadwise_count_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_k,
                                                         mean_var_count_thread_copy_step_m_k);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseWelford::Run(
                welford_mean_thread_buf(I), welford_var_thread_buf(I), welford_count_thread_buf(I));
        });

        // Step 2: do normalization and output y

        const index_t workSizePerBlock = K_BlockTileSize * num_xy_k_block_tile_iteration;

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  XYGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcYDstVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             workSizePerBlock * block_local_id +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               XYGridDesc_M_K,
                                               YElementwiseOp,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               XSrcYDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(
                    blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                    workSizePerBlock * block_local_id + thread_k_cluster_id * KThreadSliceSize),
                y_elementwise_op);

        auto threadwise_scale_load =
            ThreadwiseTensorSliceTransfer_v2<ScaleDataType,
                                             AccDataType,
                                             ScaleBiasGridDesc_M,
                                             decltype(thread_buffer_desc_m),
                                             ThreadBufferLengths_M,
                                             Sequence<0>,
                                             0,
                                             ScaleSrcVectorSize,
                                             1,
                                             true>(
                scale_grid_desc_m,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize));

        auto threadwise_bias_load = ThreadwiseTensorSliceTransfer_v2<BiasDataType,
                                                                     AccDataType,
                                                                     ScaleBiasGridDesc_M,
                                                                     decltype(thread_buffer_desc_m),
                                                                     ThreadBufferLengths_M,
                                                                     Sequence<0>,
                                                                     0,
                                                                     BiasSrcVectorSize,
                                                                     1,
                                                                     true>(
            bias_grid_desc_m,
            make_multi_index(blkgroup_id * M_BlockTileSize +
                             thread_m_cluster_id * MThreadSliceSize));

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x, x_grid_desc_m_k.GetElementSpaceSize());

        const auto scale_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_scale, scale_grid_desc_m.GetElementSpaceSize());

        const auto bias_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_bias, bias_grid_desc_m.GetElementSpaceSize());

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y, y_grid_desc_m_k.GetElementSpaceSize());

        threadwise_scale_load.Run(scale_grid_desc_m,
                                  scale_global_val_buf,
                                  thread_buffer_desc_m,
                                  make_tuple(I0),
                                  scale_thread_buf);

        threadwise_bias_load.Run(bias_grid_desc_m,
                                 bias_global_val_buf,
                                 thread_buffer_desc_m,
                                 make_tuple(I0),
                                 bias_thread_buf);

        constexpr auto xy_thread_copy_step_m_k = make_multi_index(0, K_BlockTileSize);

        for(index_t workTiles = 0; workTiles < num_xy_k_block_tile_iteration; ++workTiles)
        {
            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                AccDataType multiplier =
                    scale_thread_buf[iM] / sqrt(welford_var_thread_buf[iM] + epsilon);

                AccDataType fused_mean_bias =
                    bias_thread_buf[iM] - welford_mean_thread_buf[iM] * multiplier;

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    y_thread_buf(Number<offset>{}) =
                        x_thread_buf[Number<offset>{}] * multiplier + fused_mean_bias;
                });
            });

            threadwise_y_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   y_thread_buf,
                                   y_grid_desc_m_k,
                                   y_global_val_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, xy_thread_copy_step_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, xy_thread_copy_step_m_k);
        }

        // Step 3: update the moving average of mean and variance (optional)

        if(updateMovingAverage && block_local_id == 0 && thread_k_cluster_id == 0)
        {
            StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
                running_mean_thread_buf;
            StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
                running_var_thread_buf;

            auto running_mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultRunningMean, mean_var_grid_desc_m.GetElementSpaceSize());

            auto running_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultRunningVariance, mean_var_grid_desc_m.GetElementSpaceSize());

            auto threadwise_mean_var_load_m =
                ThreadwiseTensorSliceTransfer_v2<MeanVarDataType,
                                                 AccDataType,
                                                 MeanVarGridDesc_M,
                                                 decltype(thread_buffer_desc_m),
                                                 ThreadBufferLengths_M,
                                                 Sequence<0>,
                                                 0,
                                                 MeanVarSrcDstVectorSize,
                                                 1,
                                                 true>(
                    mean_var_grid_desc_m,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize));

            threadwise_mean_var_load_m.Run(mean_var_grid_desc_m,
                                           running_mean_global_buf,
                                           thread_buffer_desc_m,
                                           make_tuple(I0),
                                           running_mean_thread_buf);

            threadwise_mean_var_load_m.Run(mean_var_grid_desc_m,
                                           running_var_global_buf,
                                           thread_buffer_desc_m,
                                           make_tuple(I0),
                                           running_var_thread_buf);

            AccDataType oneMinusAverageFactor = type_convert<AccDataType>(1.0) - averageFactor;

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                running_mean_thread_buf(I) = running_mean_thread_buf[I] * oneMinusAverageFactor +
                                             welford_mean_thread_buf[I] * averageFactor;
                running_var_thread_buf(I) = running_var_thread_buf[I] * oneMinusAverageFactor +
                                            welford_var_thread_buf[I] * averageFactor;
            });

            auto threadwise_mean_var_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   MeanVarDataType,
                                                   decltype(thread_buffer_desc_m),
                                                   MeanVarGridDesc_M,
                                                   PassThroughOp,
                                                   ThreadBufferLengths_M,
                                                   Sequence<0>,
                                                   0,
                                                   MeanVarSrcDstVectorSize,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>(
                    mean_var_grid_desc_m,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            threadwise_mean_var_store.Run(thread_buffer_desc_m,
                                          make_tuple(I0),
                                          running_mean_thread_buf,
                                          mean_var_grid_desc_m,
                                          running_mean_global_buf);

            threadwise_mean_var_store.Run(thread_buffer_desc_m,
                                          make_tuple(I0),
                                          running_var_thread_buf,
                                          mean_var_grid_desc_m,
                                          running_var_global_buf);
        };

        // Step 4: save mean and inv-variance (optional)

        if(saveMeanInvVariance && block_local_id == 0 && thread_k_cluster_id == 0)
        {
            auto result_mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultSaveMean, mean_var_grid_desc_m.GetElementSpaceSize());

            auto result_inv_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultSaveInvVariance, mean_var_grid_desc_m.GetElementSpaceSize());

            // calculate inv-variance as 1/sqrt(epsilon+variance)
            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                welford_var_thread_buf(I) =
                    type_convert<AccDataType>(1.0f) / sqrt(epsilon + welford_var_thread_buf[I]);
            });

            auto threadwise_mean_inv_var_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   MeanVarDataType,
                                                   decltype(thread_buffer_desc_m),
                                                   MeanVarGridDesc_M,
                                                   PassThroughOp,
                                                   ThreadBufferLengths_M,
                                                   Sequence<0>,
                                                   0,
                                                   MeanVarSrcDstVectorSize,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>(
                    mean_var_grid_desc_m,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              welford_mean_thread_buf,
                                              mean_var_grid_desc_m,
                                              result_mean_global_buf);

            threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              welford_var_thread_buf,
                                              mean_var_grid_desc_m,
                                              result_inv_var_global_buf);
        };
    }
};

} // namespace ck
