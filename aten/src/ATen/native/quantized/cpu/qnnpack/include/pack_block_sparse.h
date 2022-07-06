/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <cassert>

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
  uint32_t col_block_size;  // input features block size
  uint32_t row_block_size;  // output features block size
  void print() const;
  /*
   * Unpack from BCSR to Dense
   * - Each value and zero point converted to int8_t by subtracting 128
   * - num_rows and num_cols are dimensions of dense weight tensor
   * - dst should be able to hold num_rows * num_cols elements
   * - zero_points should hold num_rows zero points
   */
  void unpack(
      int8_t* dst,
      const int64_t num_rows,
      const int64_t num_cols,
      const uint8_t* zero_points) const;
} BCSRMatrix;

std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t row_block_size,
    const uint32_t col_block_size,
    const uint8_t* zero_points);

std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const int32_t* col_indices,
    const int32_t* row_values,
    const int8_t* values,
    const int64_t col_indices_size,
    const int64_t row_values_size,
    const int64_t values_size,
    const int64_t row_block_size,
    const int64_t col_block_size);

} // namespace qnnpack
