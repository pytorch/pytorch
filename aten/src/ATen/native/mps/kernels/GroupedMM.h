#pragma once

#include <c10/metal/common.h>

// Each threadgroup computes one (16|32|64)-row by kGroupedMMTileN-column
// chunk of the output, using grouped_mm_simdgroups(rows) simdgroups.
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileN = 64;
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileK = 16;

inline constexpr uint32_t grouped_mm_simdgroups(uint32_t tile_rows) {
  return tile_rows >= 64 ? 4 : 2;
}

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
