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
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#ifndef _WIN32
#include <qnnpack/AlignedAllocator.h>
#endif
#include <qnnpack/common.h>
#include <qnnpack/math.h>

namespace qnnpack {

template <typename T>
struct OwnedOrBorrowedVector {
  using VECTOR_T =
#ifndef _WIN32
      std::vector<T, AlignedAllocator<T, 16>>;
#else
      std::vector<T>;
#endif

  std::variant<VECTOR_T, std::tuple<T*, uint32_t>> data_;

  bool owned() const {
    return (data_.index() == 0);
  }

  VECTOR_T& vector() {
    return std::get<0>(data_);
  }

  uint32_t size() const {
    if (owned()) {
      return std::get<0>(data_).size();
    } else {
      return std::get<1>(std::get<1>(data_));
    }
  }

  const T* data() const {
    if (owned()) {
      return std::get<0>(data_).data();
    } else {
      return std::get<0>(std::get<1>(data_));
    }
  }

  const T& operator[](int i) const {
    return data()[i];
  }

  OwnedOrBorrowedVector() = default;

  OwnedOrBorrowedVector(T* data_ptr, const uint32_t size) {
    data_ = std::tuple<T*, uint32_t>(data_ptr, size);
  }
};

typedef struct BCSRMatrix {
  OwnedOrBorrowedVector<uint32_t> col_indices;
  OwnedOrBorrowedVector<uint32_t> row_values;
  OwnedOrBorrowedVector<uint8_t> values;
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
    uint32_t* col_indices,
    uint32_t* row_values,
    uint8_t* values,
    const int64_t col_indices_size,
    const int64_t row_values_size,
    const int64_t values_size,
    const int64_t row_block_size,
    const int64_t col_block_size);

} // namespace qnnpack
