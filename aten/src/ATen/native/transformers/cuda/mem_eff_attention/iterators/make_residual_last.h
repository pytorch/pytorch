/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_access_iterator_residual_last.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_iterator_residual_last.h>


namespace cutlass {
namespace transform {
namespace threadblock {

template <typename BaseIterator>
struct MakeIteratorResidualLast;

template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessSize,
    Gather>> {
  using Iterator = PredicatedTileIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessSize,
      Gather>;
};

template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileAccessIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessType,
    Gather>> {
  using Iterator = PredicatedTileAccessIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType,
      Gather>;
};
} // namespace threadblock
} // namespace transform
} // namespace cutlass
