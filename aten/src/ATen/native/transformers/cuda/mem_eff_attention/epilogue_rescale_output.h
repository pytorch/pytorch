/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

  This is a copy of:
    cutlass/epilogue/thread/linear_combination.h
  (MemoryEfficientAttentionNormalize) cutlass/epilogue/threadblock/epilogue.h
  (EpilogueWithRowId) With a few modifications so that: (1) The Epilogue passes
  the row id to the OutputOp (MemoryEfficientAttentionNormalize here) Note that
  in general the fragment passed to the OutputOp could span multiple rows but it
  does not happen with the configurations we have :) (2)
  `MemoryEfficientAttentionNormalize` takes the `s_prime` and `m_prime` vectors
  (rather than scalars in `LinearCombination`) and renormalizes the output
*/

#pragma once

#include <ATen/cuda/CUDAContext.h>

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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
// output <- alpha * accumulator + beta * source
//   with:
//     alpha = 1 / s_prime (to normalize when isLast=True, 1 otherwise)
//     beta = alpha / m_prime (renormalize the output when the max changes)
//     source is the current output
template <
    typename ElementOutput_, ///< Data type used to load and store tensors
    int Count, ///< Number of elements computed per operation.
               ///< Usually it is 128/sizeof_bits<ElementOutput_>,
               ///< but we use 64 or 32 sometimes when there are not enough data
               ///< to store
    typename ElementAccumulator_, ///< Accumulator data type
    typename ElementCompute_, ///< Data type used to compute linear combination
    bool isFirst,
    bool isLast,
    typename FragmentAlphaBeta_,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class MemoryEfficientAttentionNormalize {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;
  using FragmentAlphaBeta = FragmentAlphaBeta_;

  static FloatRoundStyle const kRound = Round;

 private:
  //
  // Data members
  //

  FragmentAlphaBeta const& s_prime_;
  FragmentAlphaBeta const& m_prime_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  MemoryEfficientAttentionNormalize(
      FragmentAlphaBeta const& s_prime,
      FragmentAlphaBeta const& m_prime)
      : s_prime_(s_prime), m_prime_(m_prime) {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return !isFirst;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      int row,
      FragmentAccumulator const& accumulator,
      FragmentOutput const& source) const {
    assert(!isFirst);

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;
    ElementCompute beta = alpha * m_prime_[row];

    intermediate = mul_add_source(beta, converted_source); // X =  beta * C

    intermediate = mul_add_accumulator(
        alpha, converted_accumulator, intermediate); // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row, FragmentAccumulator const& accumulator)
      const {
    assert(isFirst);

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;

    ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;

    intermediate = mul_accumulator(
        alpha, converted_accumulator); // X =  alpha * C + uniform

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
    typename Shape_, ///< Shape of threadblock tile (concept: GemmShape)
    typename WarpMmaOperator_, ///< Warp-level MMA operator (concept:
                               ///< gemm::warp::MmaTensorOp)
    int PartitionsK, ///< Number of partitions of the K dimension
    typename OutputTileIterator_, ///< Tile iterator reading and writing output
                                  ///< tensors
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
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class EpilogueWithRowId : public EpilogueBase<
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
  EpilogueWithRowId(
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
      OutputTileIterator
          source_iterator) { ///< Threadblock tile coordinate in GEMM (in units
                             ///< of threadblock tiles)

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    } else {
      compute_source_needed_(
          output_op, destination_iterator, accumulators, source_iterator);
    }
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
      OutputTileIterator
          source_iterator ///< Threadblock tile coordinate in GEMM (in units of
                          ///< threadblock tiles)
  ) {
    typename OutputTileIterator::Fragment source_fragment;

    source_fragment.clear();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      //
      // Load the source
      //

      source_iterator.load(source_fragment);
      ++source_iterator;

      //
      // Convert and store fragment
      //

      __syncthreads();

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
          source_fragment);

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
      typename OutputTileIterator::Fragment const& source_fragment) {
    OutputAccessType* output_frag_ptr =
        reinterpret_cast<OutputAccessType*>(&output_fragment);

    AccumulatorAccessType const* compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    OutputAccessType const* source_frag_ptr =
        reinterpret_cast<OutputAccessType const*>(&source_fragment);

    int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
        OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      int row =
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess);
      // Call the output operator
      output_frag_ptr[i] =
          output_op(row, compute_frag_ptr[i], source_frag_ptr[i]);
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
      int row =
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess);
      // Call the output operator
      output_frag_ptr[i] = output_op(row, compute_frag_ptr[i]);
    }
  }

  constexpr int getRowOffset(int i) {
    using ThreadMap = typename OutputTileIterator::ThreadMap;

    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int frag_row_idx =
              (row +
               ThreadMap::Iterations::kRow *
                   (group + ThreadMap::Iterations::kGroup * cluster));
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
