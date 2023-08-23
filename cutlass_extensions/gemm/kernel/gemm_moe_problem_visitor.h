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
    \brief Scheduler for grouped GEMM
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/matrix_coord.h"

#include "cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "cutlass_extensions/gemm/kernel/moe_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/// Visitor class to abstract away the algorithm for iterating over tiles
template<typename ThreadblockShape,
         GroupScheduleMode GroupScheduleMode_,
         int               PrefetchTileCount,
         int               ThreadCount,
         bool              Transposed = false>
struct GemmMoeProblemVisitor:
    public MoeProblemVisitor<detail::GemmGroupedProblemSizeHelper<Transposed>,
                             ThreadblockShape,
                             GroupScheduleMode_,
                             PrefetchTileCount,
                             ThreadCount> {

    static bool const kTransposed = Transposed;

    using ProblemSizeHelper = detail::GemmGroupedProblemSizeHelper<Transposed>;
    using Base =
        MoeProblemVisitor<ProblemSizeHelper, ThreadblockShape, GroupScheduleMode_, PrefetchTileCount, ThreadCount>;
    using Params        = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    //
    // Methods
    //
    CUTLASS_DEVICE
    GemmMoeProblemVisitor(Params const& params_, SharedStorage& shared_storage_, int32_t block_idx):
        Base(params_, shared_storage_, block_idx)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////