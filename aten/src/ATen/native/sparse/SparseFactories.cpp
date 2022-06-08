#include <ATen/Dispatch.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sparse_coo_tensor.h>
#endif

namespace at {
namespace native {
using namespace at::sparse;
/******************************************************************************
 * Build sparse from diagonals
 ******************************************************************************/

// --------------------------------------------------------------------
// spdiags(D, O, (N,M)) -> S
//
// Take rows of D and place them on the diagonals specified by offsets O of a
// new (NxM) sparse matrix S If D is (P x Q) then O must be a row vector (P, ).
// It does not matter if Q values fit  on any diagonal of S, or if S has no
// O[i]th diagonal (those values/diagonals are simply skipped)
// --------------------------------------------------------------------

namespace {
SparseTensor _spdiags_sparse_cpu_coo(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape) {
  TensorOptions sparse_options = diagonals.options().layout(kSparse);
  TensorOptions indices_options =
      offsets.options().device(sparse_options.device());

  const int64_t n_rows_out = shape[0];
  const int64_t n_cols_out = shape[1];
  const int64_t n_cols_in = diagonals.size(1);
  const int64_t n_diag = diagonals.size(0);

  // This is the the largest number of values we could set along a diagonal, it
  // is the length of an input row or the length of the longest diagonal,
  // whichever is smaller
  const int64_t min_PQN = std::min(std::min(n_rows_out, n_cols_out), n_cols_in);
  // Conservative estimate, if all diagonals we are assigning to are of the max
  // size
  const int64_t max_nnz = n_diag * min_PQN;
  // Note: when offsets are positive we push the start point in the row of D
  // forward.
  Tensor indices = at::empty({2, max_nnz}, indices_options);
  Tensor values = at::empty({max_nnz}, diagonals.options());

  // Get accessors on input, and output
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  int64_t actual_nnz = 0;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      values.scalar_type(),
      "spdiags",
      [&] {
        auto values_accessor = values.accessor<scalar_t, 1>();
        auto const diagonals_accessor = diagonals.accessor<scalar_t, 2>();

        for (const auto d_i : c10::irange(n_diag)) {
          const int64_t& off_i = offsets_accessor[d_i];
          const int64_t row_out_begin = off_i < 0 ? std::abs(off_i) : 0;
          const int64_t col_out_begin = off_i > 0 ? off_i : 0;

          // Number of column, row positions we can assign into and the number
          // of diag positions we can read from
          const int64_t row_slots = n_rows_out - row_out_begin;
          const int64_t col_slots = n_cols_out - col_out_begin;
          const int64_t diag_slots = n_cols_in - col_out_begin;

          // Max number of reads lets us limit the loop by the first thing we
          // will exhaust, note if any of these comes up negative irange will be
          // empty so we don't have to handle that case explicitly
          const int64_t max_read =
              std::min(diag_slots, std::min(row_slots, col_slots));


          for (const auto read_count : c10::irange(max_read)) {
            const auto col = col_out_begin + read_count;
            const auto row = row_out_begin + read_count;

            values_accessor[actual_nnz] = diagonals_accessor[d_i][col];
            indices_accessor[0][actual_nnz] = row;
            indices_accessor[1][actual_nnz] = col;
            ++actual_nnz;
          }
        }
      });
  indices = indices.narrow(1, 0, actual_nnz);
  values = values.narrow(0, 0, actual_nnz);
  SparseTensor sparse =
      at::sparse_coo_tensor(indices, values, shape, sparse_options);
  return sparse;
}

// Check offsets for duplicates, and out-of-bounds diagonals
void validate_spdiags_offsets_cpu(
    const Tensor& offsets,
    const int64_t n_row,
    const int64_t n_col) {
  std::set<int64_t> seen_offsets;
  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  for (auto i : c10::irange(offsets.size(0))) {
    auto off = offsets_accessor[i];
    TORCH_CHECK(
        seen_offsets.insert(off).second,
        "Offset array contains duplicate values");
    TORCH_CHECK(
        ((-1 * n_row) < off) && (off < n_col),
        "Diagonal ",
        off,
        " does not exist in output shape (",
        n_row,
        ",",
        n_col,
        ")");
  }
}

} // namespace

SparseTensor spdiags_sparse_cpu(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout) {
  TORCH_CHECK(diagonals.dim() == 2, "Diagonals must be 2d");
  TORCH_CHECK(shape.size() == 2, "Output shape must be 2d");
  TORCH_CHECK(offsets.dim() == 1, "Offsets must be 1d");
  TORCH_CHECK(
      diagonals.size(0) == offsets.size(0),
      "Number of diagonals (",
      diagonals.size(0),
      ") does not match the number of offsets (",
      offsets.size(0),
      ")");
  if (layout) {
    TORCH_CHECK(
        (*layout == Layout::Sparse) || (*layout == Layout::SparseCsc) ||
            (*layout == Layout::SparseCsr),
        "Only output layouts (",
        Layout::Sparse,
        ", ",
        Layout::SparseCsc,
        ", and ",
        Layout::SparseCsr,
        ") are supported");
  }
  validate_spdiags_offsets_cpu(offsets, shape[0], shape[1]);

  SparseTensor result_coo = _spdiags_sparse_cpu_coo(diagonals, offsets, shape);
  if (layout) {
    if (*layout == Layout::SparseCsr) {
      return result_coo.to_sparse_csr();
    }
    if (*layout == Layout::SparseCsc) {
      return result_coo.to_sparse_csc();
    }
  }
  return result_coo;
}
} // namespace native
} // namespace at
