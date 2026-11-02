#pragma once

#include <ATen/core/Tensor.h>
#include <algorithm>
#include <limits>

namespace at::native::mps {

void metal_mm_out(const Tensor& a, const Tensor& b, const Tensor& out);

// Prefer the largest row tile that the typical group height keeps busy; the
// 48/24 cut points keep the picked tile at least 75% occupied.
inline uint32_t matmul_tile_rows(int64_t rows) {
  if (rows > 48) {
    return 64;
  }
  if (rows > 24) {
    return 32;
  }
  return 16;
}

inline bool matmul_mpp_tensor_offsets_fit(
    uint64_t rows,
    uint64_t row_stride,
    uint64_t cols,
    uint64_t col_stride) {
  constexpr auto int_max =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (rows == 0 || cols == 0) {
    return true;
  }

  // The caller bounds dimensions and strides to int32, so the sum fits in
  // uint64.
  return (rows - 1) * row_stride + (cols - 1) * col_stride <= int_max;
}

// matmul2d takes int32 extents and strides and computes offsets relative to
// each tile in int32. The kernels apply batch and tile bases in uint64 first.
template <typename Params>
inline bool matmul_mpp_indices_fit(
    const Params& params,
    uint32_t bm,
    uint32_t bn) {
  constexpr auto int_max =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (params.m > int_max || params.n > int_max || params.k > int_max ||
      params.a_stride_m > int_max || params.a_stride_k > int_max ||
      params.b_stride_k > int_max || params.b_stride_n > int_max ||
      params.out_stride_m > int_max || params.out_stride_n > int_max) {
    return false;
  }

  const auto tile_rows = std::min(params.m, bm);
  const auto tile_cols = std::min(params.n, bn);
  return matmul_mpp_tensor_offsets_fit(
             tile_rows, params.a_stride_m, params.k, params.a_stride_k) &&
      matmul_mpp_tensor_offsets_fit(
             params.k, params.b_stride_k, tile_cols, params.b_stride_n) &&
      matmul_mpp_tensor_offsets_fit(
             tile_rows, params.out_stride_m, tile_cols, params.out_stride_n);
}

} // namespace at::native::mps
