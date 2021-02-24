/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cassert>

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
  const uint32_t num_row_blocks = (N + row_block_size - 1) / row_block_size;
  // K must be > 0
  const uint32_t num_col_blocks = (K + col_block_size - 1) / col_block_size;

  bcsr_mat.row_values.reserve(num_row_blocks);
  uint32_t num_nnz_blocks{0};
  bcsr_mat.row_values.push_back(num_nnz_blocks);
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
        bcsr_mat.col_indices.push_back(j);
        num_nnz_blocks++;
        for (uint32_t ib = 0; ib < row_block_size; ++ib) {
          uint32_t row_index = i * row_block_size + ib;
          if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
            for (; row_index < (num_row_blocks * row_block_size); row_index++) {
              for (uint32_t jb = 0; jb < col_block_size; ++jb) {
                bcsr_mat.values.push_back(zero_points[N-1]);
              }
            }
            break;
          }
          for (uint32_t jb = 0; jb < col_block_size; ++jb) {
            uint32_t col_index = j * col_block_size + jb;
            if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
              bcsr_mat.values.push_back(zero_points[row_index]);
            } else {
              uint8_t val = *(a + row_index * K + col_index);
              bcsr_mat.values.push_back(val);
            }
          }
        }
      }
    }
    bcsr_mat.row_values.push_back(num_nnz_blocks);
  }
  bcsr_mat.row_block_size = row_block_size;
  bcsr_mat.col_block_size = col_block_size;
  return bcsr_mat_ptr;
}
} // namsepace qnnpack
