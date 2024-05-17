/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h>

template <typename WarpIterator>
struct TransposeWarpIterator {
  using Iterator = char;
  static bool constexpr kSupportsTranspose = false;
};

template <
    /// Operand identity
    cutlass::gemm::Operand Operand,
    /// Data type of A elements
    typename Element,
    typename InstructionShape,
    bool kTranspose>
struct TransposeWarpIterator<
    cutlass::gemm::warp::
        WarpIteratorFromSmem<Operand, Element, InstructionShape, kTranspose>> {
  using Iterator = cutlass::gemm::warp::
      WarpIteratorFromSmem<Operand, Element, InstructionShape, !kTranspose>;
  static bool constexpr kSupportsTranspose = true;
};
