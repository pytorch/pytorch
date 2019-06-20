#pragma once

#include <ATen/core/TensorOptions.h>

namespace at { namespace native {
// Different combinations of row, col, and offset can lead to two cases:
//
// Case 1 - Trapezoid (Triangle as a special case): row + offset <= col
//    Example A: offset > 0
//      1 1 0 0 0
//      1 1 1 0 0
//      1 1 1 1 0
//    Example B: offset <= 0
//      0 0 0
//      1 0 0
//      1 1 0
//    In this case, we calculate the number of elements in the first row and
//    last row of the tril respectively, and then compute the tril size.
//
// Case 2 - Trapezoid + Rectangle: row + offset > col
//    Example:
//      1 1 0
//      1 1 1
//      1 1 1
//    In this case, we first calculate the size of top trapezoid, and then
//    calculate the size of the bottom rectangle.
inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
  // number of elements in the first row of the tril
  auto m_first_row = offset > 0 ?
    std::min<int64_t>(col, 1 + offset) : // upper bounded by col
    row + offset > 0; // either 0 or 1
  // number of elements in the last row of the tril, bounded by [0, col]
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, row]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);

  // calculate # of elements in the top trapezoid
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  return tril_size;
}

inline void check_args(
    int64_t row, int64_t col, const TensorOptions& options) {
  TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
  TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
  if (options.has_layout()) {
    TORCH_CHECK(
      options.layout() == at::kStrided,
      "only support layout=torch.strided, got",
      options.layout())
  }
}

inline void check_size_nonnegative(IntArrayRef size) {
  for (auto x: size) {
    TORCH_CHECK(x >= 0, "Trying to create tensor with negative dimension ", x, ": ", size);
  }
}
} // namespace native
} // namespace at
