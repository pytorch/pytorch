#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t scaled_mm_tile = 32;
C10_METAL_CONSTEXPR uint32_t scaled_mm_simdgroups = 4;
C10_METAL_CONSTEXPR uint32_t scaled_mm_threads =
    scaled_mm_simdgroups * c10::metal::simdgroup_size;

template <typename index_t = int64_t>
struct ScaledMMParams {
  uint32_t m, n, k;
  index_t a_row_stride, a_col_stride;
  index_t b_row_stride, b_col_stride;
  index_t out_row_stride, out_col_stride;
  bool rowwise;
  bool has_bias;
  bool bias_bfloat16;
};

static_assert(
    sizeof(ScaledMMParams<>) == 72,
    "ScaledMMParams layout must match Metal");
