// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseMultiblockWelfordFirstHalf_,
          typename XDataType,
          typename MeanVarDataType,
          typename XGridDesc_M_K,
          typename MeanVarCountGridDesc_M_G,
          typename GetReduceCountPerThreadFunctor>
__global__ void kernel_multiblock_welford_first_half(
    const XGridDesc_M_K x_grid_desc_m_k,
    const MeanVarCountGridDesc_M_G mean_var_count_grid_desc_m_g,
    const GetReduceCountPerThreadFunctor get_reduce_count_per_thread,
    index_t num_k_block_tile_iteration,
    const XDataType* const __restrict__ p_x,
    MeanVarDataType* const p_welford_mean,
    MeanVarDataType* const p_welford_variance,
    int32_t* const p_welford_count)
{
    GridwiseMultiblockWelfordFirstHalf_::Run(x_grid_desc_m_k,
                                             mean_var_count_grid_desc_m_g,
                                             get_reduce_count_per_thread,
                                             num_k_block_tile_iteration,
                                             p_x,
                                             p_welford_mean,
                                             p_welford_variance,
                                             p_welford_count);
};

template <typename XDataType,
          typename AccDataType,
          typename MeanVarDataType,
          typename XGridDesc_M_K,
          typename MeanVarCountGridDesc_M_G,
          typename GetReduceCountPerThreadFunctor,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcCountSrcVectorDim,
          index_t XSrcCountSrcVectorSize>
struct GridwiseMultiblockWelfordFirstHalf
{
    static_assert((XSrcCountSrcVectorDim == 0 && MThreadSliceSize % XSrcCountSrcVectorSize == 0) ||
                      (XSrcCountSrcVectorDim == 1 &&
                       KThreadSliceSize % XSrcCountSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XSrcCountSrcVectorDim == 0);

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
                                              ThreadClusterArrangeOrder,
                                              false>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    // clang-format off
    // First half of the Multiblock Welford method to calculate mean and variance, used by both batchnorm-forward and batchnorm-backward.
    // clang-format on
    __device__ static void Run(const XGridDesc_M_K& x_grid_desc_m_k,
                               const MeanVarCountGridDesc_M_G& mean_var_count_grid_desc_m_g,
                               const GetReduceCountPerThreadFunctor& get_reduce_count_per_thread,
                               index_t num_k_block_tile_iteration,
                               const XDataType* const __restrict__ p_x,
                               MeanVarDataType* const p_welford_mean,
                               MeanVarDataType* const p_welford_variance,
                               int32_t* const p_welford_count)
    {
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            x_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            welford_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            welford_count_thread_buf;

        const index_t blkgroup_size = mean_var_count_grid_desc_m_g.GetLength(I1);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / blkgroup_size;
        const index_t block_local_id  = block_global_id % blkgroup_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K = Sequence<MThreadSliceSize, KThreadSliceSize>;
        using ThreadBufferLengths_M_1 = Sequence<MThreadSliceSize, 1>;

        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m_1 = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

        const index_t reduceSizePerBlock = K_BlockTileSize * num_k_block_tile_iteration;

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  XGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcCountSrcVectorDim,
                                                                  XSrcCountSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_welford_mean_var_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               MeanVarDataType,
                                               decltype(thread_buffer_desc_m_1),
                                               MeanVarCountGridDesc_M_G,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_1,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                mean_var_count_grid_desc_m_g,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_local_id),
                PassThroughOp{});

        auto threadwise_welford_count_store =
            ThreadwiseTensorSliceTransfer_v1r3<int32_t,
                                               int32_t,
                                               decltype(thread_buffer_desc_m_1),
                                               MeanVarCountGridDesc_M_G,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_1,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                mean_var_count_grid_desc_m_g,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_local_id),
                PassThroughOp{});

        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileSize);

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x, x_grid_desc_m_k.GetElementSpaceSize());

        auto welford_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_mean, mean_var_count_grid_desc_m_g.GetElementSpaceSize());

        auto welford_var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_variance, mean_var_count_grid_desc_m_g.GetElementSpaceSize());

        auto welford_count_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_count, mean_var_count_grid_desc_m_g.GetElementSpaceSize());

        auto threadwise_welford = ThreadwiseWelford();
        threadwise_welford.max_count_ =
            get_reduce_count_per_thread(block_local_id, thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            welford_mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            welford_var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_welford.Run(x_thread_buf, welford_mean_thread_buf, welford_var_thread_buf);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            welford_count_thread_buf(I) = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(
                welford_mean_thread_buf(I), welford_var_thread_buf(I), welford_count_thread_buf(I));
        });

        if(thread_k_cluster_id == 0)
        {
            threadwise_welford_mean_var_store.Run(thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  welford_mean_thread_buf,
                                                  mean_var_count_grid_desc_m_g,
                                                  welford_mean_global_val_buf);

            threadwise_welford_mean_var_store.Run(thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  welford_var_thread_buf,
                                                  mean_var_count_grid_desc_m_g,
                                                  welford_var_global_val_buf);

            threadwise_welford_count_store.Run(thread_buffer_desc_m_1,
                                               make_tuple(I0, I0),
                                               welford_count_thread_buf,
                                               mean_var_count_grid_desc_m_g,
                                               welford_count_global_val_buf);
        };
    }
};

} // namespace ck
