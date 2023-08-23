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
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

//#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template<
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of Scale elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    typename Enable = void>
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    bfloat16_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 80
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementScale = bfloat16_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset   = warp_idx_n * Shape::kN;
        const int quad          = lane_idx / 4;
        const int thread_offset = warp_offset + quad;
        pointer_                = smem_scales.data() + thread_offset;
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        using _MmaOperandB        = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                          == FragmentDequantizedOperand::kElements,
                      "");

        const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);

        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

            __nv_bfloat162  scalex2            = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);
            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii) {
                operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
            }
        }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
        // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
        // numerous conversion instructions in GEMM main loop.
        arch::device_breakpoint();
#endif
    }

private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Turing & Ampere
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 75
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset   = warp_idx_n * Shape::kN;
        const int quad          = lane_idx / 4;
        const int thread_offset = warp_offset + quad;
        pointer_                = smem_scales.data() + thread_offset;
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        using _MmaOperandB        = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                          == FragmentDequantizedOperand::kElements,
                      "");

        multiplies<ExpandedMmaOperandB> mul_op;

        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
        }
    }

private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x RowMajor B tensorOp, for 32x32x4 interleaved gemm
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::RowMajor>::value>::type> {

public:
    static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    // Fragment to hold scale data to apply to B before mma
    // Each 32x32x4 matmul uses 8 elements from B.
    static constexpr int ColsPerMmaTile  = 32;
    static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
    using FragmentScale                  = Array<ElementScale, TileNIterations * 8>;
    using AccessType                     = Array<ElementScale, 8>;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset   = warp_idx_n * Shape::kN;
        const int base_col      = lane_idx & 0xF8;
        const int thread_offset = warp_offset + base_col;
        pointer_                = smem_scales.data() + thread_offset;
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        AccessType* scale_frag_ptr = reinterpret_cast<AccessType*>(&scale_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
            // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
            scale_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(pointer_ + ColsPerMmaTile * tile_iter);
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        static_assert(FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");

        multiplies<FragmentDequantizedOperand> mul_op;
        operand_frag = mul_op(operand_frag, scale_frag);
    }

private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x ColumnMajor B tensorOp, for 32x32x4 interleaved gemm
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    // Fragment to hold scale data to apply to B before mma
    // Each 32x32x4 matmul uses 8 elements from B.
    static constexpr int ColsPerMmaTile  = 32;
    static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
    using FragmentScale                  = Array<ElementScale, TileNIterations * 2>;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset   = warp_idx_n * Shape::kN;
        const int base_col      = lane_idx & 0xF8 + lane_idx % 4;
        const int thread_offset = warp_offset + base_col;
        pointer_                = smem_scales.data() + thread_offset;
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
            // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
            // For col major B, each thread will jump 4 cols to get its next value inside
            // of the super mma.
            CUTLASS_PRAGMA_UNROLL
            for (int mma_iter = 0; mma_iter < 2; ++mma_iter) {
                scale_frag[tile_iter * 2 + mma_iter] = pointer_[ColsPerMmaTile * tile_iter + 4 * mma_iter];
            }
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        using MmaOperandB                 = typename ArchMmaOperator::FragmentB;
        static constexpr int total_n_mmas = 2 * TileNIterations;
        static_assert(MmaOperandB::kElements * total_n_mmas == FragmentDequantizedOperand::kElements, "");

        multiplies<MmaOperandB> mul_op;

        MmaOperandB* operand_frag_ptr = reinterpret_cast<MmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < total_n_mmas; ++mma_n_iter) {
            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
        }
    }

private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
