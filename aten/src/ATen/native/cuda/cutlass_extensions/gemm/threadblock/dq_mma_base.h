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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/mma_base.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////
// SFINAE trick so I can keep the same loop code for Volta and dispatch to the
// correct warp level mma. On volta, all data is stored to shared memory as FP16.
template<typename WarpMma, int kExpansionFactor = 1>
CUTLASS_DEVICE void run_warp_mma(WarpMma&                           warp_mma,
                                 typename WarpMma::FragmentC&       D,
                                 typename WarpMma::FragmentA const& A,
                                 typename WarpMma::FragmentB const& B,
                                 typename WarpMma::FragmentC const& C,
                                 const int                          warp_tileB_k_offset)
{
    warp_mma(D, A, B, C);
}

template<typename WarpMma, int kExpansionFactor = WarpMma::kExpansionFactor>
CUTLASS_DEVICE void run_warp_mma(WarpMma&                                      warp_mma,
                                 typename WarpMma::FragmentC&                  D,
                                 typename WarpMma::TransformedFragmentA const& A,
                                 typename WarpMma::TransformedFragmentB const& B,
                                 typename WarpMma::FragmentC const&            C,
                                 const int                                     warp_tileB_k_offset)
{
    warp_mma(D, A, B, C, warp_tileB_k_offset);
}
////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template<
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// The type of the scales
    typename ElementScale_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class DqMmaBase {
public:
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape = Shape_;

    ///< Policy describing tuning details
    using Policy = Policy_;

    ///< Type of the scale to be loaded
    using ElementScale = ElementScale_;

    //
    // Dependent types
    //

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Shape describing the overall GEMM computed from shared memory
    /// by each warp.
    using WarpGemm = typename Policy::Operator::Shape;

    /// Shape describing the number of warps filling the CTA
    using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;

    /// Number of warp-level GEMM operations
    static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

    static constexpr int kNumKIterationsPerWarpBLoad =
        Operator::IteratorB::InstructionShape::kRow / Operator::InstructionShape::kK;

    static_assert(!(kWarpGemmIterations % kNumKIterationsPerWarpBLoad), "");
    static constexpr int kWarpGemmIterationsForB = kWarpGemmIterations / kNumKIterationsPerWarpBLoad;

    /// Number of stages
    static int const kStages = Stages;

    /// Tensor reference to the A operand
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    /// Tensor reference to the B operand
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    //
    // Nested structs
    //

    /// Shared storage object needed by threadblock-scoped GEMM
    class SharedStorage {
    public:
        //
        // Type definitions
        //

        /// Shape of the A matrix operand in shared memory
        using ShapeA =
            MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow, Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

        /// Shape of the B matrix operand in shared memory
        using ShapeB =
            MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow, Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:
        //
        // Data members
        //

        /// Buffer for A operand
        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

        /// Buffer for B operand
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

        /// Buffer to hold scales for threadblock
        AlignedBuffer<ElementScale, Shape::kN> operand_scale;

    public:
        //
        // Methods
        //

        /// Returns a layout object for the A matrix
        CUTLASS_DEVICE
        static typename Operator::LayoutA LayoutA()
        {
            return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
        }

        /// Returns a layout object for the B matrix
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutB()
        {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        /// Returns a TensorRef to the A operand
        CUTLASS_HOST_DEVICE
        TensorRefA operand_A_ref()
        {
            return TensorRefA{operand_A.data(), LayoutA()};
        }

        /// Returns a TensorRef to the B operand
        CUTLASS_HOST_DEVICE
        TensorRefB operand_B_ref()
        {
            return TensorRefB{operand_B.data(), LayoutB()};
        }
    };

protected:
    //
    // Data members
    //

    /// Iterator to load a warp-scoped tile of A operand from shared memory
    typename Operator::IteratorA warp_tile_iterator_A_;

    /// Iterator to load a warp-scoped tile of B operand from shared memory
    typename Operator::IteratorB warp_tile_iterator_B_;

public:
    /// Construct from tensor references
    CUTLASS_DEVICE
    DqMmaBase(
        ///< Shared storage needed for internal use by threadblock-scoped GEMM
        SharedStorage& shared_storage,
        ///< ID within the threadblock
        int thread_idx,
        ///< ID of warp
        int warp_idx,
        ///< ID of each thread within a warp
        int lane_idx):
        warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
