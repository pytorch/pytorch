#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h>
#include <cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & bf16 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB>
struct DefaultMma<bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  bfloat16_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator,
                  false,
                  SharedMemoryClear,
                  GatherA,
                  GatherB> {

private:
    // Conversions only needed pre-ampere. This will trigger mma pipeline, so we convert before STS.
    static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
    using MmaElementA = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;
    using MmaElementB = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;

public:
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        MmaElementA,
                                                                        LayoutA,
                                                                        MmaElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        2,
                                                                        Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
        bfloat16_t,
        LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA,
        kAlignmentA,
        GatherA>;

    // Define iterators over tiles from the B operand
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
        bfloat16_t,
        LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB,
        kAlignmentB,
        GatherB>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
                                                                    IteratorA,
                                                                    typename MmaCore::SmemIteratorA,
                                                                    IteratorB,
                                                                    typename MmaCore::SmemIteratorB,
                                                                    ElementAccumulator,
                                                                    layout::RowMajor,
                                                                    typename MmaCore::MmaPolicy>;
};

// bf16 x bf16 specialization on Ampere to use mma multistage for 2 stage. Helps avoid reg spills on
// large tile when not enough shared mem is present to do 3+ stage
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB>
struct DefaultMma<bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  bfloat16_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  arch::Sm80,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator,
                  false,
                  SharedMemoryClear,
                  GatherA,
                  GatherB> {

    // Define the MmaCore components
    // 3 is used on purpose here to trigger components for mma multistage
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        bfloat16_t,
                                                                        LayoutA,
                                                                        bfloat16_t,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        3,
                                                                        Operator>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<bfloat16_t, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        bfloat16_t,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA,
        GatherA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<bfloat16_t, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        bfloat16_t,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB,
        GatherB>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape,
                                                                     IteratorA,
                                                                     typename MmaCore::SmemIteratorA,
                                                                     MmaCore::kCacheOpA,
                                                                     IteratorB,
                                                                     typename MmaCore::SmemIteratorB,
                                                                     MmaCore::kCacheOpB,
                                                                     ElementAccumulator,
                                                                     layout::RowMajor,
                                                                     typename MmaCore::MmaPolicy,
                                                                     2>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & int8 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<cutlass::bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  uint8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<cutlass::bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  uint8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      kStages,
                      Operator,
                      SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      kStages,
                      Operator,
                      SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass