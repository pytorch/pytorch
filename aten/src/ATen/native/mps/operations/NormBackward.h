#pragma once
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/safe_conv.h>

namespace at::native::mps {

// Picks and encodes the kernels in kernels/NormBackward.h that compute grad_input from a reduction over each of
// `rows` contiguous rows of `row_size` elements:
// - packed: rows of at most simdgroup_size * ILP_PER_THREAD elements, several rows per threadgroup;
// - split: too few rows to fill the GPU, or rows too long for 32-bit loop counters, are cut into chunks that a
//   row_partials pass reduces and an apply pass finishes;
// - single_row / looped: one threadgroup per row.
struct NormBackwardRows {
  NormBackwardRows(int64_t rows_, int64_t row_size_, const TensorOptions& options)
      : rows(c10::safe_conv<uint32_t>(rows_, "norm backward rows")),
        row_size(c10::safe_conv<uint32_t>(row_size_, "norm backward row size")),
        width(c10::llvm::PowerOf2Ceil(c10::metal::ceil_div(row_size, c10::metal::ILP_PER_THREAD))),
        packed(width <= c10::metal::simdgroup_size && std::in_range<uint32_t>(int64_t(rows) * width)) {
    if (packed) {
      return;
    }
    // One threadgroup per row keeps up with the split down to about a quarter of the occupancy target
    constexpr int64_t min_chunk = 16384, max_chunk = int64_t(1) << 30;
    const auto groups = occupancy_threadgroups();
    auto num_chunks = rows_ < groups / 4
        ? std::clamp<int64_t>(c10::metal::ceil_div(2 * groups, rows_), 1, c10::metal::ceil_div(row_size_, min_chunk))
        : 1;
    num_chunks = std::max(num_chunks, c10::metal::ceil_div(row_size_, max_chunk));
    chunk_size = c10::metal::round_up(c10::metal::ceil_div(row_size_, num_chunks), int64_t(c10::metal::ILP_PER_THREAD));
    chunks = c10::metal::ceil_div(row_size_, int64_t(chunk_size));
    if (chunks > 1) {
      partials = at::empty({rows_, chunks, 2}, options.dtype(kFloat));
    }
  }

  template <typename Extra>
  void encode(id<MTLComputeCommandEncoder> enc,
              MetalShaderLibrary& lib,
              std::string_view prefix,
              const Tensor& grad_out,
              const Tensor& input,
              const Tensor& stat_a,
              const Tensor& stat_b,
              const Tensor& grad_input,
              const Extra& extra) const {
    const auto type = scalarToMetalTypeString(input);
    auto pso = [&](std::string_view variant) {
      auto rc = lib.getPipelineStateForFunc(fmt::format("{}_{}_{}", prefix, variant, type));
      [enc setComputePipelineState:rc];
      return rc;
    };
    if (packed) {
      auto state = pso("packed");
      const std::array<uint32_t, 3> dims{rows, row_size, width};
      mtl_setArgs(enc, grad_out, input, stat_a, stat_b, grad_input, dims, extra);
      mtl_dispatch1DJob(enc, state, int64_t(rows) * width);
    } else if (chunks > 1) {
      const auto grid = MTLSizeMake(chunks, rows, 1), threads = MTLSizeMake(c10::metal::max_threadgroup_size, 1, 1);
      const std::array<uint32_t, 2> dims{row_size, chunk_size};
      pso("row_partials");
      mtl_setArgs(enc, grad_out, input, stat_a, stat_b, partials, dims, extra);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
      pso("apply");
      mtl_setArgs(enc, grad_out, input, stat_a, stat_b, grad_input, partials, dims, extra);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    } else {
      const auto threads =
          std::min(c10::metal::ceil_div(row_size, c10::metal::ILP_PER_THREAD), c10::metal::max_threadgroup_size);
      pso(threads * c10::metal::ILP_PER_THREAD < row_size ? "looped" : "single_row");
      mtl_setArgs(enc, grad_out, input, stat_a, stat_b, grad_input, row_size, extra);
      [enc dispatchThreadgroups:MTLSizeMake(1, rows, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
  }

  uint32_t rows, row_size, width;
  bool packed;
  uint32_t chunk_size = 0;
  int64_t chunks = 1;
  Tensor partials;
};

// Tiles a column reduction over `rows` x `cols` for kernels/NormBackward.h's norm_backward_column_tiled. Tiles are
// tx columns by ty rows. With many rows, 32x32 tiles keep 32 rows of loads in flight per column; with few rows, one
// wide row of threads keeps loads contiguous. tx is the widest power of two that leaves at most 1/8 of the lanes
// idle, and rows are split to fill the GPU with a few rows per thread.
struct NormBackwardColumns {
  NormBackwardColumns(int64_t rows, int64_t cols) {
    constexpr int64_t max_threads = c10::metal::max_threadgroup_size, min_rows_per_thread = 8;
    constexpr int64_t tall_tile = c10::metal::simdgroup_size;
    const int64_t tile_rows = rows >= tall_tile * min_rows_per_thread ? tall_tile : 1;
    tx = std::min<int64_t>(c10::llvm::PowerOf2Ceil(cols), max_threads / tile_rows);
    while (tx > c10::metal::simdgroup_size && c10::metal::round_up(cols, tx) - cols > cols / 8) {
      tx /= 2;
    }
    ty = std::min<int64_t>(c10::llvm::PowerOf2Ceil(rows), max_threads / tx);
    col_groups = c10::metal::ceil_div(cols, tx);
    splits = std::clamp<int64_t>(
        occupancy_threadgroups() / col_groups, 1, c10::metal::ceil_div(rows, ty * min_rows_per_thread));
  }

  void dispatch(id<MTLComputeCommandEncoder> enc) const {
    [enc dispatchThreadgroups:MTLSizeMake(col_groups, splits, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
  }

  int64_t tx, ty, col_groups, splits;
};

} // namespace at::native::mps
