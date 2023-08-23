/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Base scheduler for grouped problems, using MoE
*/

#pragma once

#include "cutlass/gemm/kernel/grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles
template<typename ProblemSizeHelper, typename ThreadblockShape_>
struct BaseMoeProblemVisitor {
    using ThreadblockShape = ThreadblockShape_;

    struct ProblemInfo {
        static int32_t const kNoPrefetchEntry = -1;
        int32_t              problem_idx;
        int32_t              problem_start;

        CUTLASS_DEVICE
        ProblemInfo(): problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

        CUTLASS_DEVICE
        ProblemInfo(int32_t problem_idx_, int32_t problem_start_):
            problem_idx(problem_idx_), problem_start(problem_start_)
        {
        }
    };

    struct Params {
        int64_t const* last_row_for_problem;
        int64_t        gemm_n;
        int64_t        gemm_k;
        int32_t        problem_count;
        void const*    workspace;
        int32_t        tile_count;

        //
        // Methods
        //

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params():
            last_row_for_problem(nullptr), gemm_n(0), gemm_k(0), problem_count(0), workspace(nullptr), tile_count(0)
        {
        }

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params(int64_t const* last_row_for_problem,
               int64_t        gemm_n,
               int64_t        gemm_k,
               int32_t        problem_count,
               void const*    workspace  = nullptr,
               int32_t        tile_count = 0):
            last_row_for_problem(last_row_for_problem),
            gemm_n(gemm_n),
            gemm_k(gemm_k),
            problem_count(problem_count),
            workspace(workspace),
            tile_count(tile_count)
        {
        }
    };

    Params const& params;
    int32_t       tile_idx;
    int32_t       problem_tile_start;
    int32_t       problem_idx;

    //
    // Methods
    //
    CUTLASS_DEVICE
    BaseMoeProblemVisitor(Params const& params_, int32_t block_idx):
        params(params_), tile_idx(block_idx), problem_tile_start(0), problem_idx(0)
    {
    }

    /// Get the grid shape
    CUTLASS_HOST_DEVICE
    static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem)
    {

        return cutlass::gemm::GemmCoord(((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
                                        ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN),
                                        1);
    }

    /// Gets the global tile index
    CUTLASS_HOST_DEVICE
    int32_t tile_index() const
    {
        return tile_idx;
    }

    /// Gets the index of the problem
    CUTLASS_HOST_DEVICE
    int32_t problem_index() const
    {
        return problem_idx;
    }

    CUTLASS_HOST_DEVICE
    int32_t threadblock_idx() const
    {
        return tile_idx - problem_tile_start;
    }

    CUTLASS_DEVICE
    void advance(int32_t grid_size)
    {
        tile_idx += grid_size;
    }

    CUTLASS_HOST_DEVICE
    static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem)
    {
        ProblemSizeHelper::possibly_transpose_problem(problem);
    }

    /// Returns the problem size for the current problem
    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord problem_size() const
    {
        return problem_size(problem_idx);
    }

    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord problem_size(int idx) const
    {
        const int64_t prev_problem_row    = idx == 0 ? 0 : params.last_row_for_problem[idx - 1];
        const int64_t current_problem_row = params.last_row_for_problem[idx];
        const int64_t gemm_m              = current_problem_row - prev_problem_row;
        GemmCoord problem(GemmCoord::Index(gemm_m), GemmCoord::Index(params.gemm_n), GemmCoord::Index(params.gemm_k));
        ProblemSizeHelper::possibly_transpose_problem(problem);
        return problem;
    }

    CUTLASS_HOST_DEVICE
    static int32_t tile_count(const cutlass::gemm::GemmCoord& grid)
    {
        return ProblemSizeHelper::tile_count(grid);
    }

    static int32_t group_tile_count(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count)
    {
        int32_t total_tiles = 0;
        for (int32_t i = 0; i < problem_count; ++i) {
            auto problem = host_problem_sizes_ptr[i];
            possibly_transpose_problem(problem);
            auto grid = grid_shape(problem);
            total_tiles += tile_count(grid);
        }

        return total_tiles;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ProblemSizeHelper,
         typename ThreadblockShape,
         GroupScheduleMode GroupScheduleMode_,
         int               PrefetchTileCount,
         int               ThreadCount>
struct MoeProblemVisitor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template<typename ProblemSizeHelper, typename ThreadblockShape, int PrefetchTileCount, int ThreadCount>
struct MoeProblemVisitor<ProblemSizeHelper,
                         ThreadblockShape,
                         GroupScheduleMode::kDeviceOnly,
                         PrefetchTileCount,
                         ThreadCount>: public BaseMoeProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
    using Base                                = BaseMoeProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
    using Params                              = typename Base::Params;
    static int const  kThreadCount            = ThreadCount;
    static bool const kRequiresPrecomputation = false;
    static int const  kThreadsPerWarp         = 32;

    struct SharedStorage {};

    // Final tile of the problem loaded by this thread. Each thread will hold
    // a separate value.
    int32_t problem_ending_tile;

    SharedStorage& shared_storage;

    //
    // Methods
    //
    CUTLASS_DEVICE
    MoeProblemVisitor(Params const& params_, SharedStorage& shared_storage_, int32_t block_idx):
        Base(params_, block_idx), problem_ending_tile(0), shared_storage(shared_storage_)
    {
        this->problem_idx        = -1 * kThreadsPerWarp;
        this->problem_tile_start = 0;
    }

    CUTLASS_DEVICE
    bool next_tile()
    {
        // Check whether the tile to compute is within the range of the current problem.
        int32_t problem_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, this->problem_idx % kThreadsPerWarp);
        if (this->tile_idx < problem_tile_end) {
            return true;
        }

        // Check whether the tile to compute is within the current group of problems fetched by the warp.
        // The last tile for this group is the final tile of the problem held by the final thread in the warp.
        int32_t group_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp - 1);

        // Keep the starting problem for this group in `problem_idx`. This is done to reduce
        // register pressure. The starting problem for this group is simply the first problem
        // in the group most recently fetched by the warp.
        int32_t& group_problem_start = this->problem_idx;
        group_problem_start          = (this->problem_idx / kThreadsPerWarp) * kThreadsPerWarp;

        // Keep the starting tile for this group in `problem_tile_start`. This is done to reduce
        // register pressure.
        int32_t& group_tile_start = this->problem_tile_start;

        // Each thread in the warp processes a separate problem to advance until
        // reaching a problem whose starting tile is less less than tile_idx.
        while (group_tile_end <= this->tile_idx) {
            group_problem_start += kThreadsPerWarp;
            if (group_problem_start > this->params.problem_count) {
                return false;
            }

            // Since `group_tile_start` is a reference to `this->problem_tile_start`, this
            // also sets `this->problem_tile_start`. The fact that `this->problem_tile_start`
            // is also set here is used later in `next_tile`.
            group_tile_start = group_tile_end;

            int     lane_idx     = threadIdx.x % kThreadsPerWarp;
            int32_t lane_problem = group_problem_start + lane_idx;

            // Compute the number of tiles in the problem assigned to each thread.
            problem_ending_tile = 0;
            if (lane_problem < this->params.problem_count) {
                cutlass::gemm::GemmCoord problem = this->problem_size(lane_problem);
                cutlass::gemm::GemmCoord grid    = this->grid_shape(problem);
                problem_ending_tile              = this->tile_count(grid);
            }

            // Compute a warp-wide inclusive prefix sum to compute the ending tile index of
            // each thread's problem.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
                int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
                if (lane_idx >= i) {
                    problem_ending_tile += val;
                }
            }

            // The total tile count for this group is now in the final position of the prefix sum
            int32_t tiles_in_group = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp - 1);

            problem_ending_tile += group_tile_start;
            group_tile_end += tiles_in_group;
        }

        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int32_t problem_idx_in_group = __popc(__ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));

        this->problem_idx = group_problem_start + problem_idx_in_group;

        // The starting tile for this problem is the ending tile of the previous problem. In cases
        // where `problem_idx_in_group` is the first problem in the group, we do not need to reset
        // `problem_tile_start`, because it is set to the previous group's ending tile in the while
        // loop above.
        if (problem_idx_in_group > 0) {
            this->problem_tile_start = __shfl_sync(0xffffffff, problem_ending_tile, problem_idx_in_group - 1);
        }

        return true;
    }

    static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                     int32_t                         problem_count,
                                     int32_t                         block_count)
    {
        return 0;
    }

    static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                int32_t                         problem_count,
                                int32_t                         block_count,
                                void*                           host_workspace_ptr)
    {
    }
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
