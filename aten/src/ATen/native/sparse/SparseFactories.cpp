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

void spdiags_backward_from_coo(
    const Tensor& grad_out,
    const Tensor& offsets,
    const int64_t n_diag,
    const int64_t n_col_in,
    const int64_t n_col_out,
    const int64_t n_row_out,
    Tensor& grad_in) {
  using namespace at::indexing;
  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      grad_out.scalar_type(),
      "spdiags_backward_coo",
      [&] {
        auto grad_in_accessor = grad_in.accessor<scalar_t, 2>();
        auto grad_out_values_accessor =
            get_sparse_impl(grad_out)->values_.accessor<scalar_t, 1>();
        auto grad_out_indices_accessor =
            get_sparse_impl(grad_out)->indices_.accessor<int64_t, 2>();
        auto nnz = get_sparse_impl(grad_out)->nnz();
        for (const auto i : c10::irange(n_diag)) {
          const int64_t offset_i = offsets_accessor[i];
          const int64_t abs_off = std::abs(offset_i);
          // First column index read from (grad_out), also first column index to
          // write to (grad_in/result)
          const int64_t col_begin = offset_i < 0 ? 0 : offset_i;
          // Last column index read from (grad_out) also last column index to
          // write to (grad_in/result)
          const int64_t n_col = std::min(
              n_col_in, std::min(n_row_out - abs_off, n_col_out - abs_off));
          // Loop over the range to read/write
          for (const auto j : c10::irange(n_col)) {
            const int64_t col_idx = col_begin + j;
            const int64_t row_idx = col_idx - offset_i;
            for (int64_t nnz_idx = 0; nnz_idx < nnz; ++nnz_idx) {
              if ((grad_out_indices_accessor[0][nnz_idx] == row_idx) &&
                  (grad_out_indices_accessor[1][nnz_idx] == col_idx)) {
                grad_in_accessor[i][col_idx] +=
                    grad_out_values_accessor[nnz_idx];
              }
            }
          }
        }
      });
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

Tensor spdiags_backward_sparse_cpu(
    const Tensor& grad_out,
    const Tensor& offsets,
    IntArrayRef input_shape) {
  AT_ASSERT(input_shape.size() == 2);
  AT_ASSERT(offsets.dim() == 1);
  auto n_diag = input_shape[0];
  auto n_col_in = input_shape[1];
  auto n_col_out = grad_out.size(1);
  auto n_row_out = grad_out.size(0);
  AT_ASSERT(grad_out.dim() == 2);
  AT_ASSERT(offsets.size(0) == n_diag);
  // auto output_layout = grad_out.layout();
  auto grad_in_options = grad_out.options().layout(Layout::Strided);
  Tensor grad_in = at::zeros({input_shape}, grad_in_options);
  if (grad_out.layout() == Layout::Sparse) {
    spdiags_backward_from_coo(
        grad_out, offsets, n_diag, n_col_in, n_col_out, n_row_out, grad_in);
  } else {
    // Todo, for backward efficient implementation from different formats should
    // be possible
    spdiags_backward_from_coo(
        grad_out.to_sparse(),
        offsets,
        n_diag,
        n_col_in,
        n_col_out,
        n_row_out,
        grad_in);
  }
  return grad_in;
}

} // namespace native
} // namespace at
