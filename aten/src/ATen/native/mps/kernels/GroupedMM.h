#pragma once

#include <c10/metal/common.h>

// kGroupedMMTileN and kGroupedMMTileK are used only by the
// SIMD-group-matrix kernels grouped_mm_rows and grouped_mm_k. Each threadgroup
// computes 16, 32 or 64 rows by 64 columns and advances through K in chunks
// of 16; grouped_mm_*_mpp kernels use separate template tile sizes.
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileN = 64;
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileK = 16;

inline constexpr uint32_t grouped_mm_simdgroups(uint32_t tile_rows) {
  return tile_rows >= 64 ? 4 : 2;
}

// Tuned on M5 Pro, this two/four/eight SIMD-group launch uses eight groups for
// the BN=256 tile to provide more cooperative parallelism across its wider
// output; narrower tiles reuse the row-based two/four configuration.
inline constexpr uint32_t grouped_mm_mpp_simdgroups(
    uint32_t tile_rows,
    uint32_t tile_cols) {
  return tile_cols >= 256 ? 8 : grouped_mm_simdgroups(tile_rows);
}

template <typename idx_t>
struct GroupedMMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t groups;
  idx_t a_stride_m;
  idx_t a_stride_k;
  idx_t a_batch_stride;
  idx_t b_stride_k;
  idx_t b_stride_n;
  idx_t b_batch_stride;
  idx_t out_stride_m;
  idx_t out_stride_n;
  idx_t out_batch_stride;
};
