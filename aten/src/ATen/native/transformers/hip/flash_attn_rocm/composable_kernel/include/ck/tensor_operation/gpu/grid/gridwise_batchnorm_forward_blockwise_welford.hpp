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

template <typename GridwiseBatchrNormForwardWithBlockwiseWelford_,
          typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          typename XYGridDesc_M_K,
          typename ScaleBiasGridDesc_M,
          typename MeanVarGridDesc_M,
          typename GetReduceCountPerThreadFunctor>
__global__ void kernel_batchnorm_forward_with_blockwise_welford(
    const XYGridDesc_M_K x_grid_desc_m_k,
    const XYGridDesc_M_K y_grid_desc_m_k,
    const ScaleBiasGridDesc_M scale_grid_desc_m,
    const ScaleBiasGridDesc_M bias_grid_desc_m,
    const MeanVarGridDesc_M mean_var_grid_desc_m,
    const GetReduceCountPerThreadFunctor get_reduce_count_per_thread,
    index_t num_k_block_tile_iteration,
    AccDataType epsilon,
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
    GridwiseBatchrNormForwardWithBlockwiseWelford_::Run(x_grid_desc_m_k,
                                                        y_grid_desc_m_k,
                                                        scale_grid_desc_m,
                                                        bias_grid_desc_m,
                                                        mean_var_grid_desc_m,
                                                        get_reduce_count_per_thread,
                                                        num_k_block_tile_iteration,
                                                        epsilon,
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
          typename ScaleBiasGridDesc_M,
          typename MeanVarGridDesc_M,
          typename GetReduceCountPerThreadFunctor,
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
struct GridwiseBatchNormForwardWithBlockwiseWelford
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

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<AccDataType, ThreadReduceSrcDesc_M_K, ThreadReduceDstDesc_M>;

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
                               const ScaleBiasGridDesc_M& scale_grid_desc_m,
                               const ScaleBiasGridDesc_M& bias_grid_desc_m,
                               const MeanVarGridDesc_M& mean_var_grid_desc_m,
                               const GetReduceCountPerThreadFunctor& get_reduce_count_per_thread,
                               index_t num_k_block_tile_iteration,
                               AccDataType epsilon,
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

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> scale_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> bias_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> var_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        using ThreadBufferLengths_M           = Sequence<MThreadSliceSize>;
        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

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
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
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
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
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
                make_multi_index(block_global_id * M_BlockTileSize +
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
            make_multi_index(block_global_id * M_BlockTileSize +
                             thread_m_cluster_id * MThreadSliceSize));

        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileSize);
        constexpr auto thread_copy_bwd_step_m_k = make_multi_index(0, -K_BlockTileSize);

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x, x_grid_desc_m_k.GetElementSpaceSize());

        const auto scale_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_scale, scale_grid_desc_m.GetElementSpaceSize());

        const auto bias_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_bias, bias_grid_desc_m.GetElementSpaceSize());

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y, y_grid_desc_m_k.GetElementSpaceSize());

        // Step 1:  do welford reduction to get mean and variance

        auto threadwise_welford       = ThreadwiseWelford();
        threadwise_welford.max_count_ = get_reduce_count_per_thread(thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {

            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_welford.Run(x_thread_buf, mean_thread_buf, var_thread_buf);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            int count = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(mean_thread_buf(I), var_thread_buf(I), count);
        });

        // Step 2: do normalization and output y

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

        auto thread_copy_tail_m_k = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_m_k;

        threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
        threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                AccDataType multiplier =
                    scale_thread_buf[Number<iM>{}] / sqrt(var_thread_buf[iM] + epsilon);

                AccDataType fused_mean_bias =
                    bias_thread_buf[Number<iM>{}] - mean_thread_buf[iM] * multiplier;

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // normalize
                    y_thread_buf(Number<offset>{}) =
                        x_thread_buf[Number<offset>{}] * multiplier + fused_mean_bias;
                });
            });

            threadwise_y_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   y_thread_buf,
                                   y_grid_desc_m_k,
                                   y_global_val_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_bwd_step_m_k);
        }

        // Step 3: update the moving average of mean and variance (optional)

        if(updateMovingAverage && thread_k_cluster_id == 0)
        {
            StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
                running_mean_thread_buf;
            StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
                running_var_thread_buf;

            auto running_mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultRunningMean, mean_var_grid_desc_m.GetElementSpaceSize());

            auto running_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultRunningVariance, mean_var_grid_desc_m.GetElementSpaceSize());

            auto threadwise_mean_var_load =
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
                    make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize));

            threadwise_mean_var_load.Run(mean_var_grid_desc_m,
                                         running_mean_global_buf,
                                         thread_buffer_desc_m,
                                         make_tuple(I0),
                                         running_mean_thread_buf);

            threadwise_mean_var_load.Run(mean_var_grid_desc_m,
                                         running_var_global_buf,
                                         thread_buffer_desc_m,
                                         make_tuple(I0),
                                         running_var_thread_buf);

            AccDataType oneMinusAverageFactor = type_convert<AccDataType>(1.0) - averageFactor;

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                running_mean_thread_buf(I) = running_mean_thread_buf[I] * oneMinusAverageFactor +
                                             mean_thread_buf[I] * averageFactor;
                running_var_thread_buf(I) = running_var_thread_buf[I] * oneMinusAverageFactor +
                                            var_thread_buf[I] * averageFactor;
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
                    make_multi_index(block_global_id * M_BlockTileSize +
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

        if(saveMeanInvVariance && thread_k_cluster_id == 0)
        {
            auto result_mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultSaveMean, mean_var_grid_desc_m.GetElementSpaceSize());

            auto result_inv_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                resultSaveInvVariance, mean_var_grid_desc_m.GetElementSpaceSize());

            // calculate inv-variance as 1/sqrt(epsilon+variance), stored in place of variance
            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                var_thread_buf(I) =
                    type_convert<AccDataType>(1.0f) / sqrt(epsilon + var_thread_buf[I]);
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
                    make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize),
                    PassThroughOp{});

            threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              mean_thread_buf,
                                              mean_var_grid_desc_m,
                                              result_mean_global_buf);

            threadwise_mean_inv_var_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              var_thread_buf,
                                              mean_var_grid_desc_m,
                                              result_inv_var_global_buf);
        };
    }
};

} // namespace ck
