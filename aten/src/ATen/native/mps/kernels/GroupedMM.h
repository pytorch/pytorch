#pragma once

#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t kGroupedMMTileN = 64;
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileK = 16;

// Tuned on M5 Pro: the 256-wide MPP tile wants eight SIMD groups, narrower
// tiles two or four depending on their row count.
inline constexpr uint32_t grouped_mm_simdgroups(
    uint32_t tile_rows,
    uint32_t tile_cols) {
  return tile_cols >= 256 ? 8 : tile_rows >= 64 ? 4 : 2;
}

template <typename idx_t>
struct GroupedMMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t groups;
  idx_t a_stride_m;
  idx_t a_stride_k;
  idx_t b_stride_k;
  idx_t b_stride_n;
  idx_t out_stride_m;
  idx_t out_stride_n;
  idx_t batch_stride;
};
