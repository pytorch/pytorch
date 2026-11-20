#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t scaled_mm_tile = 32;
C10_METAL_CONSTEXPR uint32_t scaled_mm_simdgroups = 4;
C10_METAL_CONSTEXPR uint32_t scaled_mm_threads =
    scaled_mm_simdgroups * c10::metal::simdgroup_size;
// K, N, storage offsets and leading strides must be multiples of this
C10_METAL_CONSTEXPR uint32_t scaled_mm_alignment = 16;
// Output tile of the Metal 4.1 kernel, which reads fp8 operands natively
C10_METAL_CONSTEXPR uint32_t scaled_mm_native_tile = 64;
// Matrix A's max number of rows on which we dispatch few rows kernel
// Higher values per-row ALU work outweighs streaming matrix B once.
C10_METAL_CONSTEXPR uint32_t scaled_mm_few_rows_max = 4;
// Number of fp8 values loaded at once by few rows kernel (one uint4).
C10_METAL_CONSTEXPR uint32_t scaled_mm_few_rows_load_bytes = 16;
// Number of K elements each simd lane handles at a time (two uint4 loads)
C10_METAL_CONSTEXPR uint32_t scaled_mm_few_rows_lane_chunk =
    scaled_mm_few_rows_load_bytes * 2;
C10_METAL_CONSTEXPR float scaled_mm_decode_scale = 256.0f;
// Values sharing a block scale in the MX (BlockWise1x32) and NVFP4
// (BlockWise1x16) recipes
C10_METAL_CONSTEXPR uint32_t scaled_mm_mx_block = 32;
C10_METAL_CONSTEXPR uint32_t scaled_mm_nvfp4_block = 16;
// K values each lane of the block-scaled few rows kernel decodes per step: one
// MXFP8 block (32 bytes) or two NVFP4 blocks (16 bytes).
C10_METAL_CONSTEXPR uint32_t scaled_mm_blockwise_chunk = 32;

template <typename index_t = int64_t>
struct ScaledMMParams {
  uint32_t m, n, k;
  index_t a_row_stride, a_col_stride;
  index_t b_row_stride, b_col_stride;
  index_t out_row_stride, out_col_stride;
  bool rowwise;
  // False for block-scaled recipes without a tensor-wise scale
  bool has_tensor_scales;
  bool has_bias;
  bool bias_bfloat16;
};

static_assert(
    sizeof(ScaledMMParams<>) == 72,
    "ScaledMMParams layout must match Metal");
