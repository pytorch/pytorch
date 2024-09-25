/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8gemm_sparse.h>
#include <requantization/runtime-sse2.h>

#include "8x4c1x4-packed-sse2.h"

// This is a super slow kernel in that it does not use intrinsics to
// tranpose. Since this is for x86 we are not optimizing it.
// For ARM this will be optimized.
void pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2(
    const size_t mr,
    const size_t K,
    const uint8_t* a,
    const size_t a_stride,
    uint8_t* a_packed) {

  // Packed A format.
  // 8kx4m blocks for alls blocks given 4 rows (4m) are placed in contiguous memory.
  // Original A
  // --------- K -----------          -- (K + 4 - 1) / 4 --
  // |                     |          |                   |
  // |                     |        (M + 8 - 1)/8         |
  // |                     | Packed   |                   |
  // M                     |  =>      |-------------------|
  // |                     |        Thus Packed A has (K + 4 - 1)/4 * (M + 8 -1)/8 blocks
  // |                     |
  // |---------------------|
  //
  // Each 8 x 4 blocks is transposed and stored.
  // Each of the (K + 4 - 1)/4 blocks for a given group of 8 m blocks
  // are stored adjacent in memory
  // Thus, each block:
  // |----8m-----|----8m-----|
  // 4k          |           | ..... (K + 4 - 1)/4 blocks
  // |-----------|-----------|
  // This locality helps in loading 8kx8m blocks of activations
  // Note when M is not multiple of 8, the rest can contain arbitrary
  // data in packed A as we will not be writing those out.
  // This wil be taken care by just copying the appropriate valid data

  // Note that parts of A that are not filled are:
  // Remainder of M blocks. So some m values are random. This is ok
  // because when sparse gemm accumulated into them, those values will not
  // be written out.
  // Remainder of K blocks. When K is not multiple of 4 the remaining k
  // in 4x8 blocks are also random. this is also ok because the packed
  // weights will be packed with zeros such that multiplication will result
  // in zero.
  uint32_t num_k_blocks = (K + COL_BLOCK_SIZE -1) / COL_BLOCK_SIZE;
  for (uint32_t k_block = 0; k_block < num_k_blocks - 1; k_block++) {
    for (uint32_t k = 0; k < COL_BLOCK_SIZE; k++) {
      for (uint32_t m = 0; m < mr; m++) {
        *(a_packed + k_block * PACKED_A_BLOCK_SIZE + k * 8 + m) =
          *(a + m * a_stride + k_block * COL_BLOCK_SIZE + k);
      }
    }
  }
  for (uint32_t k = 0; k < (K - ((num_k_blocks - 1) * COL_BLOCK_SIZE)); k++) {
    for (uint32_t m = 0; m < mr; m++) {
      *(a_packed + (num_k_blocks - 1) * PACKED_A_BLOCK_SIZE + k * 8 + m) =
        *(a + m * a_stride + (num_k_blocks - 1) * COL_BLOCK_SIZE + k);
    }
  }

}
