#pragma once

#include <c10/metal/common.h>

// Each threadgroup computes one (16|32|64)-row by kGroupedMMTileN-column
// chunk of the output, using grouped_mm_simdgroups(rows) simdgroups.
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileN = 64;
C10_METAL_CONSTEXPR uint32_t kGroupedMMTileK = 16;

inline constexpr uint32_t grouped_mm_simdgroups(uint32_t tile_rows) {
  return tile_rows >= 64 ? 4 : 2;
}

struct GroupedMMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t groups;
  uint64_t a_stride_m;
  uint64_t a_stride_k;
  uint64_t b_stride_k;
  uint64_t b_stride_n;
  uint64_t b_batch_stride;
  uint64_t out_stride_m;
  uint64_t out_stride_n;
  uint64_t out_batch_stride;
};
