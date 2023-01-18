/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Iterator over OperandA in shared memory in SIMT cases, used for
   fused-matmuls Mostly copied from cutlass/gemm/warp/mma_simt_tile_iterator.h
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/arch/memory_sm75.h>

#include <cutlass/layout/matrix.h>

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/warp/mma_simt_policy.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterates over operands to warp-level matrix multiply operations targeting
/// SIMT instructions
///
/// concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    typename Policy_>
class MmaSimtTileIteratorWithResidual;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for A operands of column-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of A elements
    typename Element_,
    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    typename Policy_>
class MmaSimtTileIteratorWithResidual<
    Shape_,
    Operand::kA,
    Element_,
    layout::ColumnMajor,
    Policy_> {
 public:
  static int const kIterations = 8;

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::ColumnMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(
      !(Shape::kRow % Policy::WarpShape::kRow),
      "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(
      Shape::kColumn > 0,
      "Shape::kColumn must be greater than zero.");
  static_assert(
      Policy::WarpShape::kRow > 0,
      "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(
      Shape::kRow / Policy::WarpShape::kRow > 0,
      "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape =
      MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  static_assert(
      !(ThreadShape::kRow % Policy::LaneMmaShape::kM),
      "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
      ThreadShape::kRow / Policy::LaneMmaShape::kM,
      ThreadShape::kColumn>;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

 private:
  /// Internal reference
  cutlass::
      TensorRef<Array<Element, Policy::LaneMmaShape::kM>, layout::ColumnMajor>
          ref_;

  /// residual access
  bool is_residual_;

  /// residual offset applied after first block
  TensorCoord residual_offset_;

  int iterations_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual() : is_residual_(false), iterations_(0) {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual(TensorRef ref, int lane_id)
      : is_residual_(false), iterations_(0) {
    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset =
        lane_layout.inverse(lane_id) * MatrixCoord(Policy::LaneMmaShape::kM, 0);

    ref.add_coord_offset(lane_offset);

    ref_.reset(
        reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(ref.data()),
        ref.stride(0) / Policy::LaneMmaShape::kM);
  }
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual(
      TensorRef ref,
      TensorCoord extent,
      int lane_id)
      : is_residual_(false), iterations_(0) {
    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset =
        lane_layout.inverse(lane_id) * MatrixCoord(Policy::LaneMmaShape::kM, 0);

    ref.add_coord_offset(lane_offset);

    ref_.reset(
        reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(ref.data()),
        ref.stride(0) / Policy::LaneMmaShape::kM);

    typename TensorCoord::Index residual_size =
        extent.column() % (kIterations * Shape::kColumn);
    if (residual_size) {
      is_residual_ = true;
      residual_offset_ =
          make_Coord(0, residual_size - kIterations * Shape::kColumn);
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual& add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual& add_tile_offset(TensorCoord const& coord) {
    ref_.add_coord_offset(
        {coord.row() * Shape::kRow / Policy::LaneMmaShape::kM,
         coord.column() * Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIteratorWithResidual& operator++() {
    ref_.add_coord_offset({0, Shape::kColumn});
    ++iterations_;

    if (iterations_ >= kIterations) {
      if (is_residual_) {
        is_residual_ = false;
        ref_.add_coord_offset(residual_offset_);
      }
      iterations_ = 0;
    }

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  /// (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
    Array<Element, Policy::LaneMmaShape::kM>* dst_ptr =
        reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {
// This logic has been replaced with calls to inline PTX to guarantee
// vectorization.
#if 0
        dst_ptr[m + k * Iterations::kRow] =
          *(ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM);
#endif

        auto ptr = ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) +
            pointer_offset / Policy::LaneMmaShape::kM;
        arch::shared_load(dst_ptr[m + k * Iterations::kRow], ptr);
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment& frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation here
  }
};
} // namespace warp
} // namespace gemm
} // namespace cutlass
