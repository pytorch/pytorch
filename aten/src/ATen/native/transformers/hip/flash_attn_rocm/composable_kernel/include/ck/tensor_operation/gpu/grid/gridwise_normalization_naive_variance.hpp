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

// Y = Normalization(X, Beta, Gamma)
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType,
          typename AccElementwiseOperation,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          bool SweepOnce>
struct GridwiseNormalizationNaiveVariance_mk_to_mk
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((YDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (YDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XSrcVectorDim == 0);

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

    using BlockwiseSumReduce = PartitionedBlockwiseReduction<AccDataType,
                                                             BlockSize,
                                                             ThreadClusterLengths_M_K,
                                                             ThreadClusterArrangeOrder,
                                                             reduce::Add,
                                                             true>;

    using ThreadwiseSumReduce = ThreadwiseReduction<AccDataType,
                                                    ThreadReduceSrcDesc_M_K,
                                                    ThreadReduceDstDesc_M,
                                                    reduce::Add,
                                                    true>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& beta_grid_desc_m_k,
                               const GridDesc_M_K& y_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               AccDataType epsilon,
                               const XDataType* const __restrict__ p_x_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const BetaDataType* const __restrict__ p_beta_global,
                               YDataType* const __restrict__ p_y_global,
                               const AccElementwiseOperation acc_elementwise_op)
    {
        if constexpr(SweepOnce)
        {
            num_k_block_tile_iteration = 1;
        }

        // LDS
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>& beta_thread_buf = gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>& x_square_thread_buf = y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            mean_square_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>& var_thread_buf =
            mean_square_thread_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I)        = reduce::Add::template GetIdentityValue<AccDataType>();
            mean_square_thread_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
        });

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             AccDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_beta_load =
            ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                             AccDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             BetaSrcVectorDim,
                                             BetaSrcVectorSize,
                                             1,
                                             true>(
                beta_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               AccElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
                acc_elementwise_op);

        // Copy x from Cache
        // one pass: fwd, second pass: bwd
        constexpr auto thread_copy_fwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : K_BlockTileSize);
        constexpr auto thread_copy_bwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_m_k.GetElementSpaceSize());

        // E(x), E[x^2], var(x)
        // FIXME: Should not hack the transform from deviceOP
        int reduce_length = x_grid_desc_m_k.GetTransforms()[I2].GetUpperLengths()[I0];

        index_t reducedTiles = 0;
        do
        {
            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                    x_square_thread_buf(Number<offset_m_k>{}) =
                        x_thread_buf(Number<offset_m_k>{}) * x_thread_buf(Number<offset_m_k>{});
                });
            });

            ThreadwiseSumReduce::Reduce(x_thread_buf, mean_thread_buf);
            ThreadwiseSumReduce::Reduce(x_square_thread_buf, mean_square_thread_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);

            ++reducedTiles;
        } while(reducedTiles < num_k_block_tile_iteration);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseSumReduce::Reduce(reduce_work_buf, mean_thread_buf(I));
            mean_thread_buf(I) = mean_thread_buf(I) / reduce_length;

            block_sync_lds();

            BlockwiseSumReduce::Reduce(reduce_work_buf, mean_square_thread_buf(I));
            mean_square_thread_buf(I) = mean_square_thread_buf(I) / reduce_length;

            // var(x) = E[x^2] - E[x]^2
            var_thread_buf(I) =
                mean_square_thread_buf(I) - (mean_thread_buf(I) * mean_thread_buf(I));
        });

        // y = (x - E[x]) / sqrt(var[x] + epsilon)
        auto thread_copy_tail_m_k = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_m_k;

        threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
        threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

        reducedTiles = 0;
        do
        {
            if constexpr(!SweepOnce)
            {
                threadwise_x_load.Run(x_grid_desc_m_k,
                                      x_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      x_thread_buf);
            }

            threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                      gamma_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      gamma_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // normalize
                    y_thread_buf(Number<offset_m_k>{}) =
                        (x_thread_buf(Number<offset_m_k>{}) - mean_thread_buf(iM)) /
                        sqrt(var_thread_buf(iM) + epsilon);

                    // gamma
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) * gamma_thread_buf(Number<offset_m_k>{});
                });
            });

            threadwise_beta_load.Run(beta_grid_desc_m_k,
                                     beta_global_val_buf,
                                     thread_buffer_desc_m_k,
                                     make_tuple(I0, I0),
                                     beta_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // beta
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) + beta_thread_buf(Number<offset_m_k>{});
                });
            });

            threadwise_y_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   y_thread_buf,
                                   y_grid_desc_m_k,
                                   y_global_val_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_bwd_step_m_k);

            ++reducedTiles;
        } while(reducedTiles < num_k_block_tile_iteration);
    }
};

} // namespace ck
