/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <qnnpack/math.h>

namespace qnnpack {

typedef struct BCSRMatrix {
  std::vector<uint32_t> col_indices;
  std::vector<uint32_t> row_values;
  std::vector<uint8_t> values;
  uint32_t col_block_size;
  void print() {
    std::cout << "row ptr\n";
    for (const auto& t : row_values) {
      std::cout << t << ", ";
    }
    std::cout << std::endl;
    std::cout << "col indices\n";
    for (const auto& t : col_indices) {
      std::cout << t << ", ";
    }
    std::cout << std::endl;
    std::cout << "Actual values\n";
    for (const auto& t : values) {
      std::cout << (uint32_t)t << ", ";
    }
    std::cout << std::endl;
  }
} BCSRMatrix;

std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t col_block_size,
    const uint8_t* zero_points);

} // namespace qnnpack
