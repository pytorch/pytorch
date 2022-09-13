// Adapted from cutlass/gemm/threadblock/default_mma_core_sm75.h
// This is very similar, except we make it work for head_dim=128.
// The original cutlass version only allows kK of the thread block to be
// at most 64. Here we set kCrosswise = max(64, ThreadblockShape::kK) instead.

/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
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
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/platform/platform.h>

#include <cutlass/numeric_types.h>
#include <cutlass/matrix_shape.h>

#include <cutlass/layout/tensor_op_multiplicand_sm75.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>

#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Operation performed by MMA
    typename Operator = cutlass::arch::OpMultiplyAdd
>
struct FMHAMmaCore;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by MMA
    typename Operator_>
struct FMHAMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                   cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor,
                   ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Operator_
                  > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisibility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Cutlass only supports Crosswise at most 64
  static int const kCrosswise = std::min(Shape::kK, 64);

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      kCrosswise / (kAccessSizeInBits / cutlass::sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      kCrosswise / (kAccessSizeInBits / cutlass::sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, kCrosswise>;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, kCrosswise>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      cutlass::layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                        kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / cutlass::sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      cutlass::layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                        kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / cutlass::sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    1,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaTensorOp,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by MMA
    typename Operator_>
struct FMHAMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                   cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_,
                   LayoutC_, cutlass::arch::OpClassTensorOp, Operator_
                  > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Cutlass only supports Crosswise at most 64
  static int const kCrosswise = std::min(Shape::kK, 64);

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      kCrosswise / (kAccessSizeInBits / cutlass::sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, kCrosswise>;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<ElementB>::value, int(128 / sizeof(ElementB))>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      cutlass::layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                        kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / cutlass::sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    cutlass::layout::PitchLinearShape<8, 4>,
    kAccessSizeInBits / cutlass::sizeof_bits<ElementB>::value
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaTensorOp,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};


} // namespace fmha
