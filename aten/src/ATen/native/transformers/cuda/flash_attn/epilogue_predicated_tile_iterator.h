// Adapted from cutlass/epilogue/threadblock/predicated_tile_iterator.h
// We just want to add the move() function, but idk how to do it without
// copying the code here.

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
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/epilogue/threadblock/output_tile_thread_map.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_params.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/transform/pitch_linear_thread_map.h>


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha {

////////////////////////////////////////////////////////////////////////////////

using namespace cutlass;
using namespace cutlass::epilogue::threadblock;

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  bool ScatterD = false,     ///< Scatter D operand or not
  bool UseCUDAStore = false
>
class EpiloguePredicatedTileIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout):
      PredicatedTileIteratorParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      )
    { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) :
      Base(base) { }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorParams params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  /// Scatter indices
  int const *indices_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  EpiloguePredicatedTileIterator(
    PredicatedTileIteratorParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    int const *indices = nullptr
  ):
    params_(params), indices_(indices)
  {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    if (ScatterD && !indices) {
      mask_.clear();
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
        LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType,
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (UseCUDAStore) {
              if (guard) {
                memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess] =
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
              }
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                  (void *)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                  guard);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) {
              byte_pointer += params_.increment_row;
            }
          }

        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) const {

    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  void move(const int step=1) {

    if (!ScatterD) {
      byte_pointer_ += step * params_.advance_row;
    }

    thread_start_row_ += step * ThreadMap::Shape::kRow;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

////////////////////////////////////////////////////////////////////////////////


} // namespace fmha
