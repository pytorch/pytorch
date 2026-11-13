#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace at::native::mps {

// Launch heuristics for the norm backward kernels in kernels/NormBackward.h.
// They are pure functions of the shape and of `groups`, the number of
// threadgroups that fills the GPU (occupancy_threadgroups()), so a pick can be
// reasoned about for any device size.

// How grad_input is computed from a reduction over each of `rows` contiguous
// rows of `row_size` elements.
enum class NormRowKernel {
  // Rows of up to 128 elements, several per threadgroup, reduced with shuffles
  Packed,
  // One threadgroup per row, held in registers
  SingleRow,
  // One threadgroup per row too long for registers, reading it twice
  Looped,
  // Rows cut into chunks: row_partials sums each chunk, then apply combines the
  // sums and writes grad_input
  Split,
};

struct NormRowConfig {
  NormRowKernel kernel = NormRowKernel::Packed;
  uint32_t rows = 0, row_size = 0;
  uint32_t lanes = 0; // threads per row
  uint32_t threads = 0; // threads per threadgroup
  uint32_t row_groups = 0; // threadgroups along the rows
  uint32_t chunks = 1; // threadgroups along each row
  uint32_t chunk_size = 0;

  // Kernel names to launch, in order, over a chunks x row_groups grid of
  // threadgroups
  std::vector<std::string_view> passes() const;
};

NormRowConfig select_norm_row_config(
    int64_t rows,
    int64_t row_size,
    int64_t groups);

// Tiling of the column sums (layer_norm weight and bias grads, weight_norm over
// the last dim) of a rows x cols matrix: threadgroups of tx columns by ty rows,
// col_groups x splits of them, with one partial sum per split.
struct NormColumnConfig {
  uint32_t tx, ty, col_groups, splits;
};

NormColumnConfig select_norm_column_config(
    int64_t rows,
    int64_t cols,
    int64_t groups);

} // namespace at::native::mps
