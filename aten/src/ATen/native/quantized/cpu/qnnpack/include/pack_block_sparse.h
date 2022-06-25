/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <qnnpack/AlignedAllocator.h>
#endif
#include <qnnpack/common.h>
#include <qnnpack/math.h>

namespace qnnpack {

typedef struct BCSRMatrix {
#ifndef _WIN32
  std::vector<uint32_t, AlignedAllocator<uint32_t, 16>> col_indices;
  std::vector<uint32_t, AlignedAllocator<uint32_t, 16>> row_values;
  std::vector<uint8_t, AlignedAllocator<uint8_t, 16>> values;
#else
  std::vector<uint32_t> col_indices;
  std::vector<uint32_t> row_values;
  std::vector<uint8_t> values;
#endif
  uint32_t col_block_size; // input features block size
  uint32_t row_block_size; // output features block size
  void print() const;
} BCSRMatrix;

std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t row_block_size,
    const uint32_t col_block_size,
    const uint8_t* zero_points);

} // namespace qnnpack
