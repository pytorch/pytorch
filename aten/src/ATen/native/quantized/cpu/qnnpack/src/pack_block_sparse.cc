/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iterator>

#include <pack_block_sparse.h>

namespace qnnpack {
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t row_block_size,
    const uint32_t col_block_size,
    const uint8_t* zero_points) {
  assert(K > 0);
  std::unique_ptr<BCSRMatrix> bcsr_mat_ptr = std::make_unique<BCSRMatrix>();
  auto& bcsr_mat = *bcsr_mat_ptr;
  auto& row_values = bcsr_mat.row_values.vector();
  auto& col_indices = bcsr_mat.col_indices.vector();
  auto& values = bcsr_mat.values.vector();

  const uint32_t num_row_blocks = (N + row_block_size - 1) / row_block_size;
  // K must be > 0
  const uint32_t num_col_blocks = (K + col_block_size - 1) / col_block_size;

  row_values.reserve(num_row_blocks);
  uint32_t num_nnz_blocks{0};
  row_values.push_back(num_nnz_blocks);
  for (uint32_t i = 0; i < num_row_blocks; ++i) {
    for (uint32_t j = 0; j < num_col_blocks; ++j) {
      bool block_zero{true};
      for (uint32_t ib = 0; ib < row_block_size; ++ib) {
        uint32_t row_index = i * row_block_size + ib;
        if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
          break;
        }
        for (uint32_t jb = 0; jb < col_block_size; ++jb) {
          uint32_t col_index = j * col_block_size + jb;
          if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
            goto block_scanned;
          }
          if (*(a + row_index * K + col_index) != zero_points[row_index]) {
            block_zero = false;
            goto block_scanned;
          }
        }
      }
block_scanned:
      if (!block_zero) {
        col_indices.push_back(j);
        num_nnz_blocks++;
        for (uint32_t ib = 0; ib < row_block_size; ++ib) {
          uint32_t row_index = i * row_block_size + ib;
          if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
            for (; row_index < (num_row_blocks * row_block_size); row_index++) {
              for (uint32_t jb = 0; jb < col_block_size; ++jb) {
                values.push_back(zero_points[N-1]);
              }
            }
            break;
          }
          for (uint32_t jb = 0; jb < col_block_size; ++jb) {
            uint32_t col_index = j * col_block_size + jb;
            if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
              values.push_back(zero_points[row_index]);
            } else {
              uint8_t val = *(a + row_index * K + col_index);
              values.push_back(val);
            }
          }
        }
      }
    }
    row_values.push_back(num_nnz_blocks);
  }
  bcsr_mat.row_block_size = row_block_size;
  bcsr_mat.col_block_size = col_block_size;
  return bcsr_mat_ptr;
}

std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    uint32_t* col_indices,
    uint32_t* row_values,
    uint8_t* values,
    const int64_t col_indices_size,
    const int64_t row_values_size,
    const int64_t values_size,
    const int64_t row_block_size,
    const int64_t col_block_size) {
  std::unique_ptr<BCSRMatrix> bcsr_mat_ptr = std::make_unique<BCSRMatrix>();
  BCSRMatrix& bcsr_mat = *bcsr_mat_ptr;
  bcsr_mat.col_indices =
      OwnedOrBorrowedVector<uint32_t>(col_indices, col_indices_size);
  bcsr_mat.row_values =
      OwnedOrBorrowedVector<uint32_t>(row_values, row_values_size);
  bcsr_mat.values = OwnedOrBorrowedVector<uint8_t>(values, values_size);
  bcsr_mat.row_block_size = row_block_size;
  bcsr_mat.col_block_size = col_block_size;
  return bcsr_mat_ptr;
}

void BCSRMatrix::print() const {
  std::cout << "row block size:" << row_block_size << std::endl;
  std::cout << "col block size:" << col_block_size << std::endl;
  std::cout << "row ptr\n";
  for (int i = 0; i < row_values.size(); i++) {
    std::cout << row_values[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "col indices\n";
  for (int i = 0; i < col_indices.size(); i++) {
    std::cout << col_indices[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "Actual values\n";
  for (int i = 0; i < values.size(); i++) {
    std::cout << (uint32_t)values[i] << ", ";
  }
  std::cout << std::endl;
}

void BCSRMatrix::unpack(
    int8_t* dst,
    const int64_t num_rows,
    const int64_t num_cols,
    const uint8_t* zero_points) const {
  for (int64_t i = 0; i < num_rows; i++) {
    memset(
        dst + i * num_cols,
        static_cast<int8_t>(static_cast<int16_t>(zero_points[i]) - 128),
        num_cols * sizeof(int8_t));
  }

  const int64_t num_block_rows = static_cast<int64_t>(row_values.size()) - 1;
  const int64_t block_size = (int64_t)row_block_size * col_block_size;
  int64_t weight_values_num = 0;
  for (int64_t block_row_num = 0; block_row_num < num_block_rows;
       block_row_num++) {
    const int64_t num_blocks_in_current_block_row =
        row_values[block_row_num + 1] - row_values[block_row_num];
    for (int64_t k = 0; k < num_blocks_in_current_block_row;
         k++) { // iterate over each block in the row
      const int64_t block_start_row_num = block_row_num * row_block_size;
      const int64_t block_start_col_num =
          (int64_t)(col_indices[weight_values_num / block_size]) *
          col_block_size;
      for (int64_t l = 0; l < block_size;
           l++) { // iterate over each value in the block
        const int64_t row_num = block_start_row_num + l / col_block_size;
        const int64_t col_num = block_start_col_num + l % col_block_size;
        if (row_num < num_rows && col_num < num_cols) {
          dst[row_num * num_cols + col_num] = static_cast<int8_t>(
              static_cast<int16_t>(values[weight_values_num]) - 128);
        }
        weight_values_num++;
      }
    }
  }
}

} // namsepace qnnpack
