#pragma once

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
#include <ATen/native/cuda/cutlass_extensions/interleaved_numeric_conversion.h>

namespace cutlass {
namespace gemm {
namespace threadblock {
////////////////////////////////////////////////////////////////////////////////

// We need to distinguish here, since we want volta support. It is too much effort
// to write shared memory iterators that are probably needed for volta to function
// properly. As a result, we allow converters both after the LDG (for volta) and after
// the LDS for Turing+.
template<
    /// Iterator for B matrix in global memory
    typename IteratorB,
    /// Warp level Mma
    typename MmaOperator,
    /// Math operation perform by warp level operator
    typename MathOperator>
struct SetConverters {
};

// Dequantize after LDG, so set transforms accordingly
template<
    /// Iterator for B matrix in global memory
    typename IteratorB,
    /// Mma Policy
    typename MmaOperator>
struct SetConverters<IteratorB, MmaOperator, arch::OpMultiplyAdd> {
    using TransformAfterLDG =
        FastInterleavedAndBiasedNumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
                                                      typename IteratorB::Element,
                                                      IteratorB::Fragment::kElements>;

    using TransformAfterLDS = NumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
                                                    typename MmaOperator::ArchMmaOperator::ElementB,
                                                    MmaOperator::FragmentB::kElements>;
};

// Dequantize after LDS, so set transforms accordingly

template<
    /// Iterator for B matrix in global memory
    typename IteratorB,
    /// Mma Policy
    typename MmaOperator>
struct SetConverters<IteratorB, MmaOperator, arch::OpMultiplyAddDequantizeInterleavedBToA> {
    using TransformAfterLDG =
        NumericArrayConverter<typename IteratorB::Element, typename IteratorB::Element, IteratorB::Fragment::kElements>;

    using TransformAfterLDS =
        FastInterleavedAndBiasedNumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
                                                      typename TransformAfterLDG::result_type::Element,
                                                      MmaOperator::FragmentB::kElements>;
};

////////////////////////////////////////////////////////////////////////////////

template<
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale_,
    /// Layout for the scale operand
    typename LayoutScale_,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    ///
    typename Enable = void>
struct DqMma;

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
