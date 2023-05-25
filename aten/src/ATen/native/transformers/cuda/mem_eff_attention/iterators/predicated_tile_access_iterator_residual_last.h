/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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
    \brief Templates calculating the address and predicates to the load of tiles
    from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile
   this iterator visits maybe partial, then the remaining tiles are complete.
   So, we only need to compute the predicates twice, once before the first tile
   and once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be
    stored in registers, and integer addition is used to advance the pointer
    through memory.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/coord.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/predicate_vector.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_view.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator_params.h>

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileAccessIteratorResidualLast
///
template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType,
    bool Gather = false>
class PredicatedTileAccessIteratorResidualLast;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for pitch-linear
/// data.
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_,
    bool Gather>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::PitchLinear,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    Gather> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  using Mask = typename UnderlyingPredicates::Mask;

  /// Uses a non-template class
  struct Params : PredicatedTileAccessIteratorParams {
    using Base = PredicatedTileAccessIteratorParams;

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : Base(
              layout.stride(0),
              MakePredicatedTileAccessIteratorDesc<
                  Shape,
                  Element,
                  Layout,
                  kAdvanceRank,
                  ThreadMap>()()) {}

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  UnderlyingPredicates the_predicates;
  Mask residual_tile_mask;

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  /// Below is used when Gather is turned on.  We need to record strided_offset
  /// and contiguous_offset seperated to compute the offset by using
  ///
  /// offset = contiguous_offset + indices[strided_offset]
  ///

  /// Gather indices
  int const* indices_;

  Index gather_offset_strided;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent, is_steady_state);
  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      /// Gather indices
      int const* indices = nullptr)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        the_predicates(extent),
        indices_(indices) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    the_predicates.get_mask(residual_tile_mask);

    // Working around a weird compiler bug happening on P100 for the backward.
    // I've seen together: the_predicates.predicates_[0] = 14 (instead of 15)
    // residual_tile_mask[0] = 15 (correct)
    //
    // Adding prints when the value is calculated (in `compute_predicates_`)
    // sometimes removes the bug. The consequence is that we skip some
    // element of a tensor, leading to wrong results
    // Setting `compute_predicates_`'s second argument (`is_steady_state`) to
    // true also seems to get rid of the bug - at the cost of twice as many
    // comparisons.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    constexpr bool kWorkAroundCompilerBug = false;
#else
    constexpr bool kWorkAroundCompilerBug = true;
#endif
    the_predicates.compute_predicates_(extent, true && !kWorkAroundCompilerBug);

    // update internal pointers
    Layout layout(params_.stride_);

    if (!Gather) {
      add_pointer_offset(layout(the_predicates.thread_offset_));
    } else {
      gather_offset_strided = the_predicates.thread_offset_.strided();
      add_pointer_offset(
          layout(make_Coord(the_predicates.thread_offset_.contiguous(), 0)));
    }
  }

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool is_residual_tile) {
    if (is_residual_tile) {
      the_predicates.set_mask(residual_tile_mask);
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    if (!Gather) {
      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
      } else {
        pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
        pointer_ += Shape::kStrided * tile_offset.strided();
      }
    } else {
      add_pointer_offset(Shape::kContiguous * tile_offset.contiguous());
      gather_offset_strided += Shape::kStrided * tile_offset.strided();
    }
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    if (Gather) {
      assert(indices_);

      if (!valid()) {
        return nullptr;
      }

      LongIndex contiguous_offset = the_predicates.iteration_contiguous_ *
              (ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value /
               8) +
          the_predicates.iteration_vector_;
      int strided_index = gather_offset_strided +
          the_predicates.iteration_strided_ * ThreadMap::Delta::kStrided;

      LongIndex strided_offset = indices_[strided_index] *
          LongIndex(params_.stride_) * sizeof_bits<Element>::value / 8;

      return reinterpret_cast<AccessType*>(
          pointer_ + contiguous_offset + strided_offset);
    }

    return reinterpret_cast<AccessType*>(
               pointer_ +
               the_predicates.iteration_contiguous_ *
                   (ThreadMap::Delta::kContiguous *
                    sizeof_bits<Element>::value) /
                   8) +
        the_predicates.iteration_vector_;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    the_predicates.operator++();

    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      if (!Gather) {
        pointer_ += params_.inc_strided_;
      }

      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    the_predicates.iteration_strided_ = 0;

    if (!Gather) {
      // advance to next tile
      pointer_ += params_.inc_next_;

      // now return to start tile - if the iterator is subsequently advanced,
      // this subtraction as well as the subsequent integer addition are both
      // elided by the compiler.
      pointer_ -= params_.inc_advance_;
    }

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    the_predicates.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    the_predicates.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for column-major
/// data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_,
    bool Gather>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::ColumnMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    Gather> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessType,
      Gather>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))){};

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.row(), extent.column()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column()),
            indices) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for row-major
/// data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_,
    bool Gather>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::RowMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    Gather> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessType,
      Gather>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))){};

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      /// Gather indices
      int const* indices = nullptr)
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.column(), extent.row()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row()),
            indices) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank 2
/// data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::AffineRankN<2>,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRankN<2>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape,
      Element,
      layout::PitchLinear,
      AdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingPredicates::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PredicatedTileAccessIteratorResidualLast;

   private:
    /// stride of pitch-linear layout (units of Element)
    Coord<Layout::kStrideRank, Layout::LongIndex> stride_;
    /// amount (in byte) to increment pointer to move to next access along
    /// contiguous dimension
    LongIndex inc_contiguous_;
    /// amount (in byte) to increment pointer from first access of current
    /// contiguous dimension to first access of next one.
    LongIndex inc_strided_;
    /// amount (in byte) to increment pointer from last access of current
    /// contiguous dimension to first access of next one.
    LongIndex inc_next_strided_;
    /// amount (in byte) to increment pointer from last access to first access
    /// of next tile
    LongIndex inc_next_;
    /// amount (in byte) to increment pointer from first access of current tile
    /// to first access of next tile
    LongIndex inc_advance_;

   public:
    // Default ctor
    CUTLASS_HOST_DEVICE
    Params()
        : stride_(0),
          inc_contiguous_(0),
          inc_strided_(0),
          inc_next_(0),
          inc_advance_(0) {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : stride_({layout.stride(0), layout.stride(1)}) {
      inc_contiguous_ =
          (LongIndex(stride_[0]) * ThreadMap::Delta::kContiguous) *
          sizeof_bits<Element>::value / 8;

      inc_strided_ = (LongIndex(stride_[1]) * ThreadMap::Delta::kStrided) *
          sizeof_bits<Element>::value / 8;

      inc_next_strided_ = inc_strided_ -
          LongIndex(ThreadMap::Iterations::kContiguous - 1) * inc_contiguous_;

      if (kAdvanceRank) {
        // advance along strided dimension
        inc_advance_ = Shape::kStrided * LongIndex(stride_[1]) *
            sizeof_bits<Element>::value / 8;
      } else {
        // advance along contiguous dimension
        inc_advance_ =
            Shape::kContiguous * stride_[0] * sizeof_bits<Element>::value / 8;
      }

      inc_next_ = inc_advance_ -
          LongIndex(ThreadMap::Iterations::kContiguous - 1) * inc_contiguous_ -
          LongIndex(ThreadMap::Iterations::kStrided - 1) * inc_strided_;
    };
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  UnderlyingPredicates the_predicates;
  Mask residual_tile_mask;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent, is_steady_state);
  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        the_predicates(extent) {
    the_predicates.set_predicates(thread_id, threadblock_offset);

    // update internal pointers
    Layout layout(params_.stride_);
    add_pointer_offset(layout(the_predicates.thread_offset_));
  }

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool is_residual_tile) {
    if (is_residual_tile) {
      the_predicates.set_mask(residual_tile_mask);
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    if (kAdvanceRank) {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset[1]);
      pointer_ += Shape::kContiguous * tile_offset[0];
    } else {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset[0]);
      pointer_ += Shape::kStrided * tile_offset[1];
    }
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(pointer_) +
        the_predicates.iteration_vector_;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    the_predicates.operator++();
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      pointer_ += params_.inc_contiguous_;
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += params_.inc_next_strided_;
      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    the_predicates.iteration_strided_ = 0;

    // advance to next tile
    pointer_ += params_.inc_next_;

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    the_predicates.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    the_predicates.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return the_predicates.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank 2
/// column-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::AffineRank2ColumnMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRank2ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  // Map to the underlying AffineRankN<2> layout
  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
      Element,
      layout::AffineRankN<2>,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given an AffineRankN<2> tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::AffineRankN<2>(layout.stride(0), layout.stride(1))){};
  };

 private:
  //
  // Data members
  //

  /// Underlying AffineRankN<2> tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.row(), extent.column()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column())) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset(
        make_Coord(tile_offset.row(), tile_offset.column()));
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank-2
/// row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::AffineRank2RowMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::AffineRank2RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  // Map to the underlying AffineRankN<2> layout
  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
      Element,
      layout::AffineRankN<2>,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given an AffineRankN<2> tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::AffineRankN<2>(layout.stride(1), layout.stride(0))){};
  };

 private:
  //
  // Data members
  //

  /// Underlying AffineRankN<2> tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.column(), extent.row()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row())) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset(
        make_Coord(tile_offset.column(), tile_offset.row()));
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for column-major
/// interleaved data. It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_,
    int InterleavedK>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::ColumnMajorInterleaved<InterleavedK>,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<
          Shape::kRow * kInterleavedK,
          Shape::kColumn / kInterleavedK>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(
                extent.row() * kInterleavedK,
                extent.column() / kInterleavedK),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.row() * kInterleavedK,
                threadblock_offset.column() / kInterleavedK)) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for row-major
/// interleaved data.
//  It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_,
    int InterleavedK>
class PredicatedTileAccessIteratorResidualLast<
    Shape_,
    Element_,
    layout::RowMajorInterleaved<InterleavedK>,
    AdvanceRank,
    ThreadMap_,
    AccessType_,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<
          Shape::kColumn * kInterleavedK,
          Shape::kRow / kInterleavedK>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(
                extent.column() * kInterleavedK,
                extent.row() / kInterleavedK),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column() * kInterleavedK,
                threadblock_offset.row() / kInterleavedK)) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
