#include <ATen/native/mps/operations/NormBackwardHeuristics.h>
#include <c10/metal/common.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/safe_conv.h>

#include <algorithm>

namespace at::native::mps {

using c10::metal::ceil_div;
constexpr int64_t ilp = c10::metal::ILP_PER_THREAD;
constexpr int64_t simd = c10::metal::simdgroup_size;
constexpr int64_t max_threads = c10::metal::max_threadgroup_size;

std::vector<std::string_view> NormRowConfig::passes() const {
  switch (kernel) {
    case NormRowKernel::Packed:
      return {"packed"};
    case NormRowKernel::SingleRow:
      return {"single_row"};
    case NormRowKernel::Looped:
      return {"looped"};
    case NormRowKernel::Split:
      return {"row_partials", "apply"};
  }
  return {};
}

NormRowConfig select_norm_row_config(int64_t rows, int64_t row_size, int64_t groups) {
  NormRowConfig cfg;
  cfg.rows = c10::safe_conv<uint32_t>(rows, "uint32_t");
  cfg.row_size = c10::safe_conv<uint32_t>(row_size, "uint32_t");
  const auto lanes = ceil_div(row_size, ilp);
  if (lanes <= simd) {
    cfg.kernel = NormRowKernel::Packed;
    cfg.lanes = c10::llvm::PowerOf2Ceil(lanes);
    cfg.threads = max_threads;
    cfg.row_groups = ceil_div(rows, max_threads / int64_t(cfg.lanes));
    return cfg;
  }
  if (lanes <= max_threads) {
    cfg.kernel = NormRowKernel::SingleRow;
    cfg.lanes = cfg.threads = lanes;
    cfg.row_groups = cfg.rows;
    return cfg;
  }
  // One threadgroup per row keeps up with the split down to about a quarter of the occupancy target. Chunks of at
  // most 2**30 elements also keep the 32-bit loop counters from wrapping.
  constexpr int64_t min_chunk = 16384, max_chunk = int64_t(1) << 30;
  auto chunks = rows < groups / 4 ? std::clamp<int64_t>(ceil_div(2 * groups, rows), 1, ceil_div(row_size, min_chunk))
                                  : 1;
  chunks = std::max(chunks, ceil_div(row_size, max_chunk));
  cfg.chunk_size = c10::metal::round_up(ceil_div(row_size, chunks), ilp);
  cfg.chunks = ceil_div(row_size, int64_t(cfg.chunk_size));
  cfg.kernel = cfg.chunks > 1 ? NormRowKernel::Split : NormRowKernel::Looped;
  cfg.lanes = cfg.threads = max_threads;
  cfg.row_groups = cfg.rows;
  return cfg;
}

// With many rows, 32x32 tiles keep 32 rows of loads in flight per column; with few rows, one wide row of threads
// keeps loads contiguous. tx is the widest power of two that leaves at most 1/8 of the lanes idle, and rows are
// split to fill the GPU with a few rows per thread.
NormColumnConfig select_norm_column_config(int64_t rows, int64_t cols, int64_t groups) {
  constexpr int64_t min_rows_per_thread = 8;
  const int64_t tile_rows = rows >= simd * min_rows_per_thread ? simd : 1;
  int64_t tx = std::min(static_cast<int64_t>(c10::llvm::PowerOf2Ceil(cols)), max_threads / tile_rows);
  while (tx > simd && c10::metal::round_up(cols, tx) - cols > cols / 8) {
    tx /= 2;
  }
  const auto ty = std::min(static_cast<int64_t>(c10::llvm::PowerOf2Ceil(rows)), max_threads / tx);
  const auto col_groups = ceil_div(cols, tx);
  const auto splits = std::clamp<int64_t>(groups / col_groups, 1, ceil_div(rows, ty * min_rows_per_thread));
  return {static_cast<uint32_t>(tx),
          static_cast<uint32_t>(ty),
          c10::safe_conv<uint32_t>(col_groups, "uint32_t"),
          static_cast<uint32_t>(splits)};
}

} // namespace at::native::mps
