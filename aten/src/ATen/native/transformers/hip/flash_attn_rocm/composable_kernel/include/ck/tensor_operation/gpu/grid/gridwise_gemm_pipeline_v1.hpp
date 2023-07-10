// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"

namespace ck {

template <index_t NumPrefetch>
struct GridwiseGemmPipeline_v1;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<1>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// 2-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<2>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        {
            // Read 0
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

            // Move
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            // Read 1
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);
        }

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Write i
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

                // Read i+2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                // Sync
                block_sync_lds();

                // Gemm i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

                // Read i+3
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                // Sync
                block_sync_lds();

                // Gemm i+1
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            // Write num_loop - 2
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            // Sync
            block_sync_lds();

            // Write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <index_t NumPrefetch>
struct GridwiseGemmPipelineInterwave_v1;

template <>
struct GridwiseGemmPipelineInterwave_v1<1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // block_sync_lds(); // moved into blockwise_gemm

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// Note: 2 stage prefetch not optimized for inter-wave loop scheduler
template <>
struct GridwiseGemmPipelineInterwave_v1<2> : public GridwiseGemmPipeline_v1<2>
{
};

// TODO: deprecate as GridwiseGemmPipeline_Selector covers the functionality
template <index_t NumPrefetch, LoopScheduler LoopSched>
constexpr auto GridwiseGemmPipeline_v1_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return GridwiseGemmPipeline_v1<NumPrefetch>{};
    }
    else if constexpr(LoopSched == LoopScheduler::Interwave)
    {
        return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
    }
}

} // namespace ck
