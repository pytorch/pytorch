// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t K_BlockTileSize, index_t KThreadSliceSize>
struct GetReduceCountPerThreadForBlockwiseWelford
{
    GetReduceCountPerThreadForBlockwiseWelford(index_t numBlockTileIteration,
                                               long_index_t reduce_length)
        : numBlockTileIteration_{numBlockTileIteration}
    {
        count_in_last_tile_ = reduce_length % K_BlockTileSize;
    };

    __device__ index_t operator()(index_t thread_k_cluster_id) const
    {
        if(count_in_last_tile_ == 0)
            return (KThreadSliceSize * numBlockTileIteration_);
        else
        {
            index_t num_complete_slice  = count_in_last_tile_ / KThreadSliceSize;
            index_t count_in_last_slice = count_in_last_tile_ % KThreadSliceSize;

            if(thread_k_cluster_id < num_complete_slice)
                return (KThreadSliceSize * numBlockTileIteration_);
            else if(thread_k_cluster_id == num_complete_slice)
                return (KThreadSliceSize * (numBlockTileIteration_ - 1) + count_in_last_slice);
            else
                return (KThreadSliceSize * (numBlockTileIteration_ - 1));
        };
    };

    index_t numBlockTileIteration_;
    index_t count_in_last_tile_;
};

template <index_t K_BlockTileSize, index_t KThreadSliceSize>
struct GetReduceCountPerThreadForMultiblockWelford
{
    GetReduceCountPerThreadForMultiblockWelford(index_t blkGroupSize,
                                                index_t numBlockTileIteration,
                                                long_index_t reduce_length)
        : blkGroupSize_(blkGroupSize), numBlockTileIteration_{numBlockTileIteration}
    {
        last_block_reduce_length_ =
            reduce_length - K_BlockTileSize * numBlockTileIteration_ * (blkGroupSize_ - 1);
        numBlockTileIterationByLastBlock_ =
            (last_block_reduce_length_ + K_BlockTileSize - 1) / K_BlockTileSize;
    };

    __device__ index_t operator()(index_t block_local_id, index_t thread_k_cluster_id) const
    {
        if(last_block_reduce_length_ == K_BlockTileSize * numBlockTileIteration_ ||
           block_local_id < blkGroupSize_ - 1)
            return (KThreadSliceSize * numBlockTileIteration_);

        index_t count_in_last_tile = last_block_reduce_length_ % K_BlockTileSize;

        if(count_in_last_tile == 0)
            return (KThreadSliceSize * numBlockTileIterationByLastBlock_);
        else
        {
            index_t num_complete_slice = count_in_last_tile / KThreadSliceSize;

            if(thread_k_cluster_id < num_complete_slice)
                return (KThreadSliceSize * numBlockTileIterationByLastBlock_);
            else if(thread_k_cluster_id == num_complete_slice)
                return (KThreadSliceSize * (numBlockTileIterationByLastBlock_ - 1) +
                        count_in_last_tile);
            else
                return (KThreadSliceSize * (numBlockTileIterationByLastBlock_ - 1));
        };
    };

    index_t blkGroupSize_;
    index_t numBlockTileIteration_;

    index_t last_block_reduce_length_;
    index_t numBlockTileIterationByLastBlock_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
