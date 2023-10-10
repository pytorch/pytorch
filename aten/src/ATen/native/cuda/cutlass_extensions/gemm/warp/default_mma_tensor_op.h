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
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

#include <cutlass_extensions/arch/mma.h>
#include <cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h>

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template<
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements,
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_,
                          InstructionShape_,
                          ElementA,
                          LayoutA,
                          ElementB,
                          LayoutB,
                          ElementC,
                          LayoutC,
                          arch::OpMultiplyAddDequantizeInterleavedBToA,
                          PartitionsK,
                          AccumulatorsInRowMajor> {

private:
    // Shape for computing the FP16s
    using ComputeInstructionShape = InstructionShape_;

    // Chosen so we get K=16 for int8 and K=32 for int4.
    static constexpr int LoadInstructionK = 8 * sizeof_bits<ElementA>::value / sizeof_bits<ElementB>::value;

    // Shape for loading the narrow data type from shared memory
    using LoadInstructionShape = GemmShape<InstructionShape_::kM, InstructionShape_::kN, LoadInstructionK>;

public:
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<InstructionShape_,
                                                                             32,
                                                                             ElementA,
                                                                             cutlass::layout::RowMajor,
                                                                             ElementA,
                                                                             cutlass::layout::ColumnMajor,
                                                                             ElementC,
                                                                             cutlass::layout::RowMajor,
                                                                             arch::OpMultiplyAdd>,
                                                          cutlass::MatrixShape<1, 1>>;

    // Define the warp-level tensor op
    using Type = cutlass::gemm::warp::MmaTensorOpComputeBWithF16<WarpShape_,
                                                                 ElementA,
                                                                 LayoutA,
                                                                 ElementB,
                                                                 LayoutB,
                                                                 ElementC,
                                                                 LayoutC,
                                                                 Policy,
                                                                 LoadInstructionShape,
                                                                 PartitionsK,
                                                                 AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
