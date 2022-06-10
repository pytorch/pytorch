#include <ATen/Dispatch.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <numeric>
#include "ATen/TensorIterator.h"
#include "c10/core/Scalar.h"
#include "c10/util/Exception.h"

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
    IntArrayRef shape,
    const Tensor& nnz_per_diag,
    const Tensor& nnz_prefix,
    const int64_t nnz) {
  TensorOptions sparse_options = diagonals.options().layout(kSparse);
  TensorOptions indices_options = offsets.options();

  const int64_t n_rows_out = shape[0];
  const int64_t n_cols_out = shape[1];
  const int64_t n_diag = diagonals.size(0);

  Tensor indices = at::empty({2, nnz}, indices_options);
  Tensor values = at::empty({nnz}, diagonals.options());

  // The maximum value anything in the indices tensor can take is the largest
  // dim of the output shape
  auto index_max = std::max(n_rows_out, n_cols_out);
  Tensor idx_tmp = at::arange(index_max, indices_options);

  auto index_write_ptr = indices.data_ptr<int64_t>();
  auto nnz_prefix_ptr = nnz_prefix.data_ptr<int64_t>();
  auto nnz_per_diag_ptr = nnz_per_diag.data_ptr<int64_t>();
  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      values.scalar_type(),
      "_spdiags_sparse_cpu_coo",
      [&] {
        // Get accessor on input, output are contiguous
        auto const diagonals_accessor = diagonals.accessor<scalar_t, 2>();
        auto values_write_ptr = values.data_ptr<scalar_t>();
        at::parallel_for(
            0,
            n_diag,
            0,
            [&](const int64_t begin_diag, const int64_t end_diag) {
              auto thread_nnz_prefix = nnz_prefix_ptr[begin_diag];
              auto t_values_write_ptr = values_write_ptr + thread_nnz_prefix;
              auto t_row_index_write_ptr = index_write_ptr + thread_nnz_prefix;
              auto t_col_index_write_ptr = t_row_index_write_ptr + nnz;

              for (auto i = begin_diag; i < end_diag; ++i) {
                const int64_t off_i = offsets_accessor[i];
                const int64_t col_out_begin = std::max<int64_t>(off_i, 0);
                const int64_t row_out_begin = col_out_begin - off_i;
                auto nnz_i = nnz_per_diag_ptr[i];
                for (const auto j : c10::irange(nnz_i)) {
                  t_values_write_ptr[j] =
                      diagonals_accessor[i][col_out_begin + j];
                }
                // inc value write ptr
                t_values_write_ptr += nnz_i;

                // Select start of index read buffer for row indices
                auto row_idx_read_ptr =
                    idx_tmp.data_ptr<int64_t>() + row_out_begin;
                t_row_index_write_ptr = std::copy(
                    row_idx_read_ptr,
                    row_idx_read_ptr + nnz_i,
                    t_row_index_write_ptr);

                // Select start of index read buffer for col indices
                auto col_idx_read_ptr =
                    idx_tmp.data_ptr<int64_t>() + col_out_begin;
                t_col_index_write_ptr = std::copy(
                    col_idx_read_ptr,
                    col_idx_read_ptr + nnz_i,
                    t_col_index_write_ptr);
              }
            });
      });

  SparseTensor sparse =
      at::sparse_coo_tensor(indices, values, shape, sparse_options);
  return sparse;
}

// Check offsets for duplicates, and out-of-bounds diagonals
// While checking offsets, compute nnz per diagonal array, prefix sum array, and
// nnz total for later use
int64_t precompute_nnz_and_validate_offsets_cpu(
    const Tensor& offsets,
    const int64_t n_row_out,
    const int64_t n_col_out,
    const int64_t n_col_in,
    Tensor& nnz_per_diag,
    Tensor& nnz_prefix) {
  std::set<int64_t> seen_offsets;
  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  auto nnz_per_diag_accessor = nnz_per_diag.accessor<int64_t, 1>();
  auto nnz_prefix_accessor = nnz_prefix.accessor<int64_t, 1>();
  int64_t nnz_sum = 0;
  for (auto i : c10::irange(offsets.size(0))) {
    auto off = offsets_accessor[i];
    TORCH_CHECK(
        seen_offsets.insert(off).second,
        "spdiags(): Offset array contains duplicate values");
    TORCH_CHECK(
        ((-1 * n_row_out) < off) && (off < n_col_out),
        "spdiags(): Diagonal ",
        off,
        " does not exist in output shape (",
        n_row_out,
        ",",
        n_col_out,
        "). Valid offsets for this shape: [",
        (-n_row_out) + 1,
        ",",
        n_col_out - 1,
        "]");
    if (off >= 0) {
      nnz_per_diag_accessor[i] = std::max<int64_t>(
          std::min(std::min(n_col_out - off, n_row_out), n_col_in - off), 0);
    } else {
      nnz_per_diag_accessor[i] = std::max<int64_t>(
          std::min(std::min(n_col_out, n_row_out + off), n_col_in), 0);
    }
    nnz_prefix_accessor[i] = nnz_sum;
    nnz_sum += nnz_per_diag_accessor[i];
  }
  return nnz_sum;
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
  auto nnz_per_diag = at::empty_like(offsets);
  auto nnz_prefix = at::empty_like(offsets);
  int64_t nnz = precompute_nnz_and_validate_offsets_cpu(
      offsets, shape[0], shape[1], diagonals.size(1), nnz_per_diag, nnz_prefix);

  SparseTensor result_coo = _spdiags_sparse_cpu_coo(
      diagonals, offsets, shape, nnz_per_diag, nnz_prefix, nnz);
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
