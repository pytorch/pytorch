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
    \brief Instantiates the right WarpIterator to read from shared memory
    The class `DefaultWarpIteratorAFromSharedMemory` is useful when reading
        data dumped with `B2bGemm::accumToSmem`.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h>
#include <cutlass/platform/platform.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    typename WarpShape,
    typename InstructionShape,
    typename RegularWarpIterator,
    typename Policy,
    typename Enable = void>
struct DefaultWarpIteratorAFromSharedMemory {};

// TensorOp - Ampere half
template <typename RegularWarpIterator, typename Policy, int kInstrK>
struct DefaultWarpIteratorAFromSharedMemory<
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, kInstrK>,
    RegularWarpIterator,
    Policy,
    typename platform::enable_if<(
        sizeof_bits<typename RegularWarpIterator::Element>::value == 16 &&
        Policy::Operator::Policy::OpDelta::kRow == 1)>::type> {
  using OpDelta = typename Policy::Operator::Policy::OpDelta;
  using WarpShape = cutlass::MatrixShape<32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, kInstrK>;

  using WarpIterator = cutlass::gemm::warp::WarpIteratorFromSmem<
      cutlass::gemm::Operand::kA,
      typename RegularWarpIterator::Element,
      cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>>;
};

// TensorOp - Ampere f32
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 8>,
    RegularWarpIterator,
    Policy,
    typename platform::enable_if<(
        sizeof_bits<typename RegularWarpIterator::Element>::value != 16 ||
        Policy::Operator::Policy::OpDelta::kRow != 1)>::type> {
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr auto kWarpSize = 32;
  using OpDelta = typename Policy::Operator::Policy::OpDelta;

  using WarpIterator =
      cutlass::gemm::warp::MmaTensorOpMultiplicandTileAccessIterator<
          cutlass::MatrixShape<WarpShape::kM, WarpShape::kK>,
          cutlass::gemm::Operand::kA,
          typename RegularWarpIterator::Element,
          cutlass::layout::RowMajor,
          cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>,
          OpDelta::kRow,
          kWarpSize>;
};

// TensorOp - Volta
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<
    WarpShape,
    cutlass::gemm::GemmShape<16, 16, 4>,
    RegularWarpIterator,
    Policy> {
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 4>;
  static constexpr auto kWarpSize = 32;
  using OpDelta = typename Policy::Operator::Policy::OpDelta;

  using WarpIterator =
      cutlass::gemm::warp::MmaVoltaTensorOpMultiplicandTileIterator<
          cutlass::MatrixShape<32, 32>, // MatrixShape<WarpShape::kM,
                                        // WarpShape::kK>,
          cutlass::gemm::Operand::kA,
          typename RegularWarpIterator::Element,
          cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>,
          cutlass::MatrixShape<16, 4>,
          OpDelta::kRow,
          kWarpSize>;
};

// Simt
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<
    WarpShape,
    cutlass::gemm::GemmShape<1, 1, 1>,
    RegularWarpIterator,
    Policy> {
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr auto kWarpSize = 32;

  // We just use the same iterator, as we reproduced the same shared-memory
  // schema. Just modify it to handle non-complete tiles.
  using WarpIterator = RegularWarpIterator;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
