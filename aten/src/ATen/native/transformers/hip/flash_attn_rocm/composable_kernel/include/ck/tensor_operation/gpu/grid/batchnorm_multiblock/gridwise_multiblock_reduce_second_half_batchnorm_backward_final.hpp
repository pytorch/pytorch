// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseReduceSecondHalfBatchNormBackwardFinal_,
          typename XDataType,
          typename DyDataType,
          typename DxDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          typename XYGridDesc_M_K,
          typename DscaleDbiasGridDesc_M_K,
          typename MeanVarGridDesc_M,
          typename ScaleBiasGridDesc_M>
__global__ void kernel_reduce_second_half_batchnorm_backward_final(
    const XYGridDesc_M_K x_grid_desc_m_k,
    const XYGridDesc_M_K dy_grid_desc_m_k,
    const XYGridDesc_M_K dx_grid_desc_m_k,
    const DscaleDbiasGridDesc_M_K dscale_dbias_grid_desc_m_k,
    const MeanVarGridDesc_M mean_var_grid_desc_m,
    const ScaleBiasGridDesc_M scale_grid_desc_m,
    const ScaleBiasGridDesc_M bias_grid_desc_m,
    index_t blkgroup_size,
    long_index_t reduce_size,
    index_t num_xy_k_block_tile_iteration,
    index_t num_dscale_dbias_k_block_tile_iteration,
    const DscaleDbiasDataType* const __restrict__ p_reduce_dscale,
    const DscaleDbiasDataType* const __restrict__ p_reduce_dbias,
    const MeanVarDataType* const __restrict__ p_mean,
    const MeanVarDataType* const __restrict__ p_inv_var,
    const XDataType* const __restrict__ p_x,
    const DyDataType* const __restrict__ p_dy,
    const ScaleDataType* const __restrict__ p_scale,
    const DyElementwiseOp dy_elementwise_op,
    DxDataType* const __restrict__ p_dx,
    DscaleDbiasDataType* const __restrict__ p_dscale,
    DscaleDbiasDataType* const __restrict__ p_dbias)
{
    GridwiseReduceSecondHalfBatchNormBackwardFinal_::Run(x_grid_desc_m_k,
                                                         dy_grid_desc_m_k,
                                                         dx_grid_desc_m_k,
                                                         dscale_dbias_grid_desc_m_k,
                                                         mean_var_grid_desc_m,
                                                         scale_grid_desc_m,
                                                         bias_grid_desc_m,
                                                         blkgroup_size,
                                                         reduce_size,
                                                         num_xy_k_block_tile_iteration,
                                                         num_dscale_dbias_k_block_tile_iteration,
                                                         p_reduce_dscale,
                                                         p_reduce_dbias,
                                                         p_mean,
                                                         p_inv_var,
                                                         p_x,
                                                         p_dy,
                                                         p_scale,
                                                         dy_elementwise_op,
                                                         p_dx,
                                                         p_dscale,
                                                         p_dbias);
};

template <typename XDataType,
          typename DyDataType,
          typename DxDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          typename XYGridDesc_M_K,
          typename DscaleDbiasGridDesc_M_K,
          typename MeanVarGridDesc_M,
          typename ScaleBiasGridDesc_M,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XDyDxVectorDim,
          index_t XSrcVectorSize,
          index_t DySrcVectorSize,
          index_t DxDstVectorSize,
          index_t ScaleSrcVectorSize,
          index_t DscaleDbiasDstVectorSize,
          index_t MeanVarSrcVectorSize>
struct GridwiseReduceSecondHalfBatchNormBackwardFinal
{
    static_assert((XDyDxVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0 &&
                   MThreadSliceSize % DySrcVectorSize == 0 &&
                   MThreadSliceSize % DxDstVectorSize == 0) ||
                      (XDyDxVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0 &&
                       KThreadSliceSize % DySrcVectorSize == 0 &&
                       KThreadSliceSize % DxDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XDyDxVectorDim == 0);

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

    using BlockwiseReduce = PartitionedBlockwiseReduction<AccDataType,
                                                          BlockSize,
                                                          ThreadClusterLengths_M_K,
                                                          ThreadClusterArrangeOrder,
                                                          ck::reduce::Add,
                                                          false>;

    using ThreadwiseReduce = ThreadwiseReduction<AccDataType,
                                                 ThreadReduceSrcDesc_M_1,
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
    // Step 1: Second half of Reduction: dbias = sum(dy), dscale = sum(dy * (x-mean) * inv-variance)
    // Step 2: calculating dx = 1/reduce_size * inv-variance * scale * (reduce_size * dy - dbias - dscale * (x - mean) * inv-variance)) elementwise-ly
    // clang-format on
    __device__ static void Run(const XYGridDesc_M_K& x_grid_desc_m_k,
                               const XYGridDesc_M_K& dy_grid_desc_m_k,
                               const XYGridDesc_M_K& dx_grid_desc_m_k,
                               const DscaleDbiasGridDesc_M_K& dscale_dbias_grid_desc_m_k,
                               const MeanVarGridDesc_M& mean_var_grid_desc_m,
                               const ScaleBiasGridDesc_M& scale_grid_desc_m,
                               const ScaleBiasGridDesc_M& dscale_dbias_grid_desc_m,
                               index_t blkgroup_size,
                               long_index_t reduce_size,
                               index_t num_xy_k_block_tile_iteration,
                               index_t num_dscale_dbias_k_block_tile_iteration,
                               const DscaleDbiasDataType* const __restrict__ p_reduce_dscale,
                               const DscaleDbiasDataType* const __restrict__ p_reduce_dbias,
                               const MeanVarDataType* const __restrict__ p_mean,
                               const MeanVarDataType* const __restrict__ p_inv_var,
                               const XDataType* const __restrict__ p_x,
                               const DyDataType* const __restrict__ p_dy,
                               const ScaleDataType* const __restrict__ p_scale,
                               const DyElementwiseOp dy_elementwise_op,
                               DxDataType* const __restrict__ p_dx,
                               DscaleDbiasDataType* const __restrict__ p_dscale,
                               DscaleDbiasDataType* const __restrict__ p_dbias)
    {
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * 1, true>
            reduce_dscale_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * 1, true>
            reduce_dbias_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> dscale_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> dbias_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            dy_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            dx_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            inv_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> scale_thread_buf;

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
        // Step 1: do final reduction of dbias = sum(dy), dscale = sum(dy * (x-mean) * inv-variance)
        // clang-format on

        auto threadwise_dscale_dbias_load_m_k =
            ThreadwiseTensorSliceTransfer_v2<DscaleDbiasDataType,
                                             AccDataType,
                                             DscaleDbiasGridDesc_M_K,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             Sequence<0, 1>,
                                             1,
                                             1,
                                             1,
                                             true>(
                dscale_dbias_grid_desc_m_k,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * 1));

        auto threadwise_dscale_dbias_store_m =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               DscaleDbiasDataType,
                                               decltype(thread_buffer_desc_m),
                                               ScaleBiasGridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,
                                               0,
                                               DscaleDbiasDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                dscale_dbias_grid_desc_m,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        const auto reduce_dscale_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_reduce_dscale, dscale_dbias_grid_desc_m_k.GetElementSpaceSize());

        const auto reduce_dbias_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_reduce_dbias, dscale_dbias_grid_desc_m_k.GetElementSpaceSize());

        auto dscale_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dscale, dscale_dbias_grid_desc_m.GetElementSpaceSize());

        auto dbias_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dbias, dscale_dbias_grid_desc_m.GetElementSpaceSize());

        constexpr auto dscale_dbias_thread_copy_step_m_k =
            make_multi_index(0, KThreadClusterSize * 1);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            dscale_thread_buf(I) = type_convert<AccDataType>(0.0f);
            dbias_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_dscale_dbias_k_block_tile_iteration;
            ++reducedTiles)
        {
            threadwise_dscale_dbias_load_m_k.Run(dscale_dbias_grid_desc_m_k,
                                                 reduce_dscale_global_buf,
                                                 thread_buffer_desc_m_1,
                                                 make_tuple(I0, I0),
                                                 reduce_dscale_thread_buf);

            threadwise_dscale_dbias_load_m_k.Run(dscale_dbias_grid_desc_m_k,
                                                 reduce_dbias_global_buf,
                                                 thread_buffer_desc_m_1,
                                                 make_tuple(I0, I0),
                                                 reduce_dbias_thread_buf);

            ThreadwiseReduce::Reduce(reduce_dscale_thread_buf, dscale_thread_buf);
            ThreadwiseReduce::Reduce(reduce_dbias_thread_buf, dbias_thread_buf);

            threadwise_dscale_dbias_load_m_k.MoveSrcSliceWindow(dscale_dbias_grid_desc_m_k,
                                                                dscale_dbias_thread_copy_step_m_k);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseReduce::Reduce(reduce_work_buf, dscale_thread_buf(I));
            block_sync_lds();
            BlockwiseReduce::Reduce(reduce_work_buf, dbias_thread_buf(I));
        });

        threadwise_dscale_dbias_store_m.Run(thread_buffer_desc_m,
                                            make_tuple(I0),
                                            dscale_thread_buf,
                                            dscale_dbias_grid_desc_m,
                                            dscale_global_buf);

        threadwise_dscale_dbias_store_m.Run(thread_buffer_desc_m,
                                            make_tuple(I0),
                                            dbias_thread_buf,
                                            dscale_dbias_grid_desc_m,
                                            dbias_global_buf);

        // clang-format off
        // Step 2: calculate dx = 1/N * inv-variance * scale * (N * dy - dbias - dscale * (x - mean) * inv-variance)
        // clang-format on

        const index_t workSizePerBlock = K_BlockTileSize * num_xy_k_block_tile_iteration;

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  XYGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XDyDxVectorDim,
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
                                                                   XDyDxVectorDim,
                                                                   DySrcVectorSize,
                                                                   1,
                                                                   true>(
            dy_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             workSizePerBlock * block_local_id +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dx_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               DxDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               XYGridDesc_M_K,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               XDyDxVectorDim,
                                               DxDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                dx_grid_desc_m_k,
                make_multi_index(
                    blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                    workSizePerBlock * block_local_id + thread_k_cluster_id * KThreadSliceSize),
                PassThroughOp{});

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

        auto threadwise_mean_var_load =
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

        const auto x_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x, x_grid_desc_m_k.GetElementSpaceSize());

        const auto dy_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dy, dy_grid_desc_m_k.GetElementSpaceSize());

        auto dx_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dx, dx_grid_desc_m_k.GetElementSpaceSize());

        const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_scale, scale_grid_desc_m.GetElementSpaceSize());

        const auto mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean, mean_var_grid_desc_m.GetElementSpaceSize());

        const auto inv_var_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_inv_var, mean_var_grid_desc_m.GetElementSpaceSize());

        threadwise_scale_load.Run(scale_grid_desc_m,
                                  scale_global_buf,
                                  thread_buffer_desc_m,
                                  make_tuple(I0),
                                  scale_thread_buf);

        threadwise_mean_var_load.Run(mean_var_grid_desc_m,
                                     mean_global_buf,
                                     thread_buffer_desc_m,
                                     make_tuple(I0),
                                     mean_thread_buf);

        threadwise_mean_var_load.Run(mean_var_grid_desc_m,
                                     inv_var_global_buf,
                                     thread_buffer_desc_m,
                                     make_tuple(I0),
                                     inv_var_thread_buf);

        constexpr auto xy_thread_copy_step_m_k = make_multi_index(0, K_BlockTileSize);

        AccDataType inv_reduce_size =
            type_convert<AccDataType>(1.0) / type_convert<AccDataType>(reduce_size);

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
                AccDataType multiplier =
                    inv_reduce_size * inv_var_thread_buf[iM] * scale_thread_buf[iM];

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    dy_elementwise_op(dy_thread_buf(Number<offset>{}),
                                      dy_thread_buf[Number<offset>{}]);

                    AccDataType norm_x = (x_thread_buf[Number<offset>{}] - mean_thread_buf[iM]) *
                                         inv_var_thread_buf[iM];

                    AccDataType tmpVal = norm_x * dscale_thread_buf[iM];

                    dx_thread_buf(Number<offset>{}) =
                        multiplier *
                        (type_convert<AccDataType>(reduce_size) * dy_thread_buf[Number<offset>{}] -
                         dbias_thread_buf[iM] - tmpVal);
                });
            });

            threadwise_dx_store.Run(thread_buffer_desc_m_k,
                                    make_tuple(I0, I0),
                                    dx_thread_buf,
                                    dx_grid_desc_m_k,
                                    dx_global_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, xy_thread_copy_step_m_k);
            threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, xy_thread_copy_step_m_k);
            threadwise_dx_store.MoveDstSliceWindow(dx_grid_desc_m_k, xy_thread_copy_step_m_k);
        }
    };
};

} // namespace ck
