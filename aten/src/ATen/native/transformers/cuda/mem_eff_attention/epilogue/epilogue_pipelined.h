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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  File copied from <cutlass/epilogue/threadblock/epilogue.h>
  then modified to:
  (1) load 2 source fragments at the same time (pipelining)
  (2) support reading from a different dtype
  (3) pass the row id to the OutputOp if it takes it
    (see MemoryEfficientAttentionNormalize)
  Note that in general the fragment passed to the OutputOp could
  span multiple rows but it does not happen with the configurations we have
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <cassert>
#endif

#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_coord.h>

#include <cutlass/gemm/gemm.h>

#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/regular_tile_iterator.h>

#include <cutlass/epilogue/threadblock/epilogue_base.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <typename Op>
struct ApplyEpilogueOp {
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum,
      typename Op::FragmentOutput const& source) {
    return output_op(accum, source);
  }
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum) {
    return output_op(accum);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
    typename Shape_, ///< Shape of threadblock tile (concept: GemmShape)
    typename WarpMmaOperator_, ///< Warp-level MMA operator (concept:
                               ///< gemm::warp::MmaTensorOp)
    int PartitionsK, ///< Number of partitions of the K dimension
    typename OutputTileIterator_, ///< Tile iterator writing output tensors
    typename AccumulatorFragmentIterator_, ///< Fragment iterator selecting
                                           ///< accumulators
    typename WarpTileIterator_, ///< Warp-scoped tile iterator writing
                                ///< accumulators to SMEM
    typename SharedLoadIterator_, ///< Threadblock-scoped tile iterator loading
                                  ///< from SMEM
    typename OutputOp_, ///< Output operator
    typename Padding_, ///< Padding added to SMEM allocation to avoid bank
                       ///< conflicts (concept: MatrixShape)
    int FragmentsPerPartition =
        1, ///< Used to coarsten the epilogue granularity
    int IterationsUnroll = ///< Used to reduce binary size when epilogue op is
                           ///< large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value),
    typename OutputTileSourceIterator_ =
        OutputTileIterator_ ///< Tile iterator reading tensors
    >
class EpiloguePipelined : public EpilogueBase<
                              Shape_,
                              typename WarpMmaOperator_::Shape,
                              PartitionsK,
                              AccumulatorFragmentIterator_,
                              WarpTileIterator_,
                              Padding_,
                              FragmentsPerPartition> {
 public:
  using Base = EpilogueBase<
      Shape_,
      typename WarpMmaOperator_::Shape,
      PartitionsK,
      AccumulatorFragmentIterator_,
      WarpTileIterator_,
      Padding_,
      FragmentsPerPartition>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using OutputTileSourceIterator = OutputTileSourceIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;
  using ElementSource = typename OutputTileSourceIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef =
      typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<
      typename OutputTileIterator::Element,
      OutputTileIterator::kElementsPerAccess>;
  using SourceAccessType = Array<
      typename OutputTileSourceIterator::Element,
      OutputTileSourceIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<
      typename WarpTileIterator::Element,
      OutputTileIterator::kElementsPerAccess>;

  /// Number of warps
  using WarpCount = typename Base::WarpCount;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1
      ? Base::kFragmentsPerIteration
      : kPartitionsK;
  static int constexpr kSmemPointerOffset =
      Base::SharedStorage::StorageShape::kCount / kSmemTiles;

 public:
  static_assert(
      OutputTileSourceIterator::Fragment::kElements ==
          OutputTileIterator::Fragment::kElements,
      "Mismatch between input tile and output tile iterator (kElements)");
  static_assert(
      OutputTileSourceIterator::kIterations == OutputTileIterator::kIterations,
      "Mismatch between input tile and output tile iterator (kIterations)");
  static_assert(
      SharedLoadIterator::Fragment::kElements ==
          OutputTileIterator::Fragment::kElements,
      "Mismatch between shared load iterator and output tile iterator.");

  static_assert(
      OutputTileIterator::kElementsPerAccess,
      "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(
      !(OutputTileIterator::Fragment::kElements %
        OutputTileIterator::kElementsPerAccess),
      "Divisibility");

 private:
  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

 public:
  /// Constructor
  CUTLASS_DEVICE
  EpiloguePipelined(
      typename Base::SharedStorage& shared_storage, ///< Shared storage object
      int thread_idx, ///< ID of a thread within the threadblock
      int warp_idx, ///< ID of warp within threadblock
      int lane_idx ///< Id of thread within warp
      )
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {}

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
      OutputOp const& output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const&
          accumulators, ///< Complete warp-level accumulator tile
      OutputTileSourceIterator
          source_iterator) { ///< Threadblock tile coordinate in GEMM (in units
                             ///< of threadblock tiles)

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    } else {
      compute_source_needed_(
          output_op, destination_iterator, accumulators, source_iterator);
    }
  }
  CUTLASS_DEVICE
  void operator()(
      OutputOp const& output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const&
          accumulators) { ///< Complete warp-level accumulator tile
    compute_source_not_needed_(output_op, destination_iterator, accumulators);
  }

 private:
  template <class Seq>
  struct acc2smem_source_not_needed;

  template <size_t... Seq>
  struct acc2smem_source_not_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(
        AccumulatorFragmentIterator accum_fragment_iterator,
        WarpTileIterator& warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);
        ++accum_fragment_iterator;

        warp_tile_iterator.store(accum_fragment);
        if (p < Base::kFragmentsPerIteration - 1) {
          warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        warp_tile_iterator.add_pointer_offset(
            kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }

    CUTLASS_DEVICE
    static void push(
        size_t pos,
        AccumulatorFragmentIterator const& iterator_begin,
        WarpTileIterator& warp_tile_iterator) {
      int dummy[] = {
          (pos == (Seq * Base::kFragmentsPerIteration)) &&
          (helper<Seq * Base::kFragmentsPerIteration>(
               iterator_begin, warp_tile_iterator),
           0)...};

      CUTLASS_UNUSED(dummy[0]);
    }
  };

  static_assert(
      kPartitionsK == 1 || Base::kFragmentsPerIteration == 1,
      "One of these must be exactly 1.");

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
      OutputOp const& output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const&
          accumulators ///< Complete warp-level accumulator tile
  ) {
    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

#pragma unroll(                                                          \
    IterationsUnroll                                                     \
        ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration \
        : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations;
         iter += Base::kFragmentsPerIteration) {
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem_source_not_needed<cutlass::make_index_sequence<
          OutputTileIterator::kIterations / Base::kFragmentsPerIteration>>::
          push(iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename SharedLoadIterator::Fragment
            aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);

        if (p < Base::kFragmentsPerIteration - 1) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        } else if (kPartitionsK > 1) {
          plus<typename SharedLoadIterator::Fragment> add_fragments;

          CUTLASS_PRAGMA_UNROLL
          for (int i = 1; i < kPartitionsK; ++i) {
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            shared_load_iterator_.load(aligned_accum_fragment[i]);
            aligned_accum_fragment[0] = add_fragments(
                aligned_accum_fragment[0], aligned_accum_fragment[i]);
          }

          shared_load_iterator_.add_pointer_offset(
              (1 - kPartitionsK) * kSmemPointerOffset);
        }

        //
        // Compute the output result
        //

        typename OutputTileIterator::Fragment output_fragment;

        apply_output_operator_source_not_needed_(
            destination_iterator.thread_start_row(),
            output_fragment,
            output_op,
            aligned_accum_fragment[0]);

        //
        // Store the final result
        //

        destination_iterator.store(output_fragment);
        ++destination_iterator;
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(
            kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }
  }

  template <class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(
        AccumulatorFragmentIterator accum_fragment_iterator,
        WarpTileIterator& warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(
        size_t pos,
        AccumulatorFragmentIterator const& iterator_begin,
        WarpTileIterator& warp_tile_iterator) {
      int dummy[] = {
          (pos == Seq) &&
          (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
      OutputOp const& output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const&
          accumulators, ///< Complete warp-level accumulator tile
      OutputTileSourceIterator
          source_iterator ///< Threadblock tile coordinate in GEMM (in units of
                          ///< threadblock tiles)
  ) {
    typename OutputTileSourceIterator::Fragment source_fragment[2];

    source_fragment[0].clear();
    source_iterator.load(source_fragment[0]);
    ++source_iterator;
    source_fragment[1].clear();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      if (iter > 0) {
        __syncthreads();
      }
      //
      // Load the source for next iteration (pipelining)
      //

      if (iter + 1 < OutputTileIterator::kIterations) {
        source_iterator.load(source_fragment[(iter + 1) % 2]);
      }
      ++source_iterator;
      acc2smem_source_needed<
          cutlass::make_index_sequence<OutputTileIterator::kIterations>>::
          push(iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment
          aligned_accum_fragment[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // If the number of k-slices is > 1 - perform a reduction amongst the
      // k-slices
      if (kPartitionsK > 1) {
        plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(
              aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment;

      apply_output_operator_(
          destination_iterator.thread_start_row(),
          output_fragment,
          output_op,
          aligned_accum_fragment[0],
          source_fragment[iter % 2]);

      //
      // Store the final result
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
      int begin_row,
      typename OutputTileIterator::Fragment& output_fragment,
      OutputOp const& output_op, ///< Output operator
      typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
      typename OutputTileSourceIterator::Fragment const& source_fragment) {
    OutputAccessType* output_frag_ptr =
        reinterpret_cast<OutputAccessType*>(&output_fragment);

    AccumulatorAccessType const* compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    SourceAccessType const* source_frag_ptr =
        reinterpret_cast<SourceAccessType const*>(&source_fragment);

    int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
        OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // Call the output operator
      output_frag_ptr[i] = ApplyEpilogueOp<OutputOp>::apply(
          output_op,
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess),
          compute_frag_ptr[i],
          source_frag_ptr[i]);
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
      int begin_row,
      typename OutputTileIterator::Fragment& output_fragment,
      OutputOp const& output_op, ///< Output operator
      typename SharedLoadIterator::Fragment const& aligned_accum_fragment) {
    OutputAccessType* output_frag_ptr =
        reinterpret_cast<OutputAccessType*>(&output_fragment);

    AccumulatorAccessType const* compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
        OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // Call the output operator
      output_frag_ptr[i] = ApplyEpilogueOp<OutputOp>::apply(
          output_op,
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess),
          compute_frag_ptr[i]);
    }
  }

  // This should be constexpr, but it's only supported on c++14
  constexpr int CUTLASS_HOST_DEVICE getRowOffset(int i) {
    using ThreadMap = typename OutputTileIterator::ThreadMap;

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int frag_row_idx =
              (row +
               ThreadMap::Iterations::kRow *
                   (group + ThreadMap::Iterations::kGroup * cluster));
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int frag_idx = ThreadMap::kElementsPerAccess *
                (frag_row_idx * ThreadMap::Iterations::kColumn + column);
            if (i < frag_idx + ThreadMap::kElementsPerAccess) {
              return row_offset;
            }
          }
        }
      }
    }
    return -1;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
