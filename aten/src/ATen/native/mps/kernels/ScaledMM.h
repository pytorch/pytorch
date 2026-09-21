#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t scaled_mm_tile = 32;
C10_METAL_CONSTEXPR uint32_t scaled_mm_simdgroups = 4;
C10_METAL_CONSTEXPR uint32_t scaled_mm_threads =
    scaled_mm_simdgroups * c10::metal::simdgroup_size;
// Matrix A's max number of rows on which we dispatch gemv kernel
// Higher values per-row ALU work outweighs streaming matrix B once.
C10_METAL_CONSTEXPR uint32_t scaled_mm_gemv_max_rows = 4;
// Number of fp8 values loaded at once by gemv kernel (one uint4).
// K, leading strides and storage offsets must be multiples of this.
C10_METAL_CONSTEXPR uint32_t scaled_mm_gemv_load_bytes = 16;
// Number of K elements each simd lane handles at a time (two uint4 loads)
C10_METAL_CONSTEXPR uint32_t scaled_mm_gemv_lane_chunk =
    scaled_mm_gemv_load_bytes * 2;
C10_METAL_CONSTEXPR float scaled_mm_decode_scale = 256.0f;

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
