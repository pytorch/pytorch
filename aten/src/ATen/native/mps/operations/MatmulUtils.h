#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/mps/kernels/Matmul.h>
#include <c10/util/TypeCast.h>
#include <algorithm>
#include <limits>

namespace at::native::mps {

// out = a * b for float32, float16 and bfloat16. out must not overlap a or b.
void metal_mm_out(const Tensor& a, const Tensor& b, const Tensor& out);

template <typename idx_t>
MatmulParams<idx_t> matmul_params(
    const Tensor& a,
    const Tensor& b,
    const Tensor& out) {
  return {
      .m = c10::checked_convert<uint32_t>(a.size(-2), "a.size(-2)"),
      .n = c10::checked_convert<uint32_t>(b.size(-1), "b.size(-1)"),
      .k = c10::checked_convert<uint32_t>(
          std::min(a.size(-1), b.size(-2)), "contraction dimension"),
      .a_stride_m = static_cast<idx_t>(a.stride(-2)),
      .a_stride_k = static_cast<idx_t>(a.stride(-1)),
      .b_stride_k = static_cast<idx_t>(b.stride(-2)),
      .b_stride_n = static_cast<idx_t>(b.stride(-1)),
      .out_stride_m = static_cast<idx_t>(out.stride(-2)),
      .out_stride_n = static_cast<idx_t>(out.stride(-1)),
  };
}

// Prefer the largest row tile that `rows` keeps at least 75% occupied.
inline uint32_t matmul_tile_rows(int64_t rows) {
  const auto keeps_occupied = [rows](int64_t tile_rows) {
    return rows > tile_rows * 3 / 4;
  };
  if (keeps_occupied(kMatmulTileRowsLarge)) {
    return kMatmulTileRowsLarge;
  }
  if (keeps_occupied(kMatmulTileRowsMedium)) {
    return kMatmulTileRowsMedium;
  }
  return kMatmulTileRowsSmall;
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
inline bool matmul_mpp_indices_fit(
    const MatmulParams<uint64_t>& params,
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
