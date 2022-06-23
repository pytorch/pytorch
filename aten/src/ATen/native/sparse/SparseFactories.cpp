#include <ATen/Dispatch.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/sparse/SparseFactories.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

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
void _spdiags_kernel_cpu(
    TensorIterator& iter,
    const Tensor& diagonals,
    Tensor& values,
    Tensor& indices) {
  auto* row_index_write_ptr = indices[0].data_ptr<int64_t>();
  auto* col_index_write_ptr = indices[1].data_ptr<int64_t>();
  const int64_t diagonals_read_stride = diagonals.stride(1);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      diagonals.scalar_type(),
      "spdiags_cpu",
      [&] {
        auto* values_write_ptr = values.data_ptr<scalar_t>();
        cpu_kernel(
            iter,
            [&](int64_t diag_index,
                int64_t diag_offset,
                int64_t out_offset,
                int64_t n_out) -> int64_t {
              auto* rows_start = row_index_write_ptr + out_offset;
              auto* cols_start = col_index_write_ptr + out_offset;
              auto* vals_start = values_write_ptr + out_offset;
              int64_t const first_col = std::max<int64_t>(diag_offset, 0);
              int64_t const first_row = first_col - diag_offset;
              auto* data_read = diagonals[diag_index].data_ptr<scalar_t>() +
                  first_col * diagonals_read_stride;
              for (int64_t i = 0; i < n_out; ++i) {
                rows_start[i] = first_row + i;
                cols_start[i] = first_col + i;
                vals_start[i] = data_read[i * diagonals_read_stride];
              }
              // dummy return
              return 0;
            });
      });
}

// Check offsets for duplicates, and out-of-bounds diagonals
// While checking offsets, compute nnz per diagonal
void _spdiags_setup_cpu(
    TensorIterator& iter,
    int64_t n_row_out,
    int64_t n_col_in,
    int64_t n_col_out) {
  const int64_t min_col_in_out = std::min(n_col_in, n_col_out);

  cpu_kernel(iter, [&](int64_t offset) {
    TORCH_CHECK(
        ((-1 * n_row_out) < offset) && (offset < n_col_out),
        "spdiags(): Diagonal ",
        offset,
        " does not exist in output shape (",
        n_row_out,
        ",",
        n_col_out,
        "). Valid offsets for this shape: [",
        (-n_row_out) + 1,
        ",",
        n_col_out - 1,
        "]");
    if (offset >= 0) {
      return std::max<int64_t>(std::min(min_col_in_out - offset, n_row_out), 0);
    } else {
      return std::max<int64_t>(std::min(min_col_in_out, n_row_out + offset), 0);
    }
  });
}

void _spdiags_backward_kernel_cpu(
    TensorIterator& iter,
    const int64_t n_diag,
    const int64_t offsets_stride,
    IntArrayRef grad_in_strides,
    IntArrayRef grad_in_shape) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      iter.dtype(),
      "spdiags_backward_cpu",
      [&] {
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          scalar_t* grad_in_data = reinterpret_cast<scalar_t*>(data[0]);
          int64_t* offsets_data = reinterpret_cast<int64_t*>(data[1]);
          // Search offsets for the given offset value and return the index,
          // this is the row position in grad_in
          auto find_row_idx = [&offsets_data, &offsets_stride, &n_diag](
                                  int64_t offset) -> int64_t {
            for (int64_t idx = 0; idx < n_diag; ++idx) {
              if (offsets_data[idx * offsets_stride] == offset) {
                return idx;
              }
            }
            return -1;
          };

          auto* grad_out_values_bytes = data[2];
          auto* grad_out_row_indices_bytes = data[3];
          auto* grad_out_col_indices_bytes = data[4];

          for (int64_t i = 0; i < n; ++i) {
            auto value = *reinterpret_cast<scalar_t*>(grad_out_values_bytes);
            auto row_out_idx =
                *reinterpret_cast<int64_t*>(grad_out_row_indices_bytes);
            auto col_idx =
                *reinterpret_cast<int64_t*>(grad_out_col_indices_bytes);
            auto comp_offset = col_idx - row_out_idx;
            auto row_idx = find_row_idx(comp_offset);
            if ((row_idx >= 0) && (row_idx < grad_in_shape[0]) &&
                (col_idx < grad_in_shape[1])) {
              grad_in_data
                  [row_idx * grad_in_strides[0] +
                   col_idx * grad_in_strides[1]] = value;
            }
            // Update nnz length inputs, grad_in/offsets are restrided and do
            // not update
            grad_out_values_bytes += strides[2];
            grad_out_row_indices_bytes += strides[3];
            grad_out_col_indices_bytes += strides[4];
          }
        };
        iter.for_each(loop);
      });
}
} // namespace

Tensor spdiags_cpu(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout) {
  return impl::spdiags_impl(
      diagonals,
      offsets,
      shape,
      layout,
      _spdiags_setup_cpu,
      _spdiags_kernel_cpu);
}

Tensor spdiags_backward_cpu(
    const Tensor& grad_out,
    const Tensor& offsets,
    IntArrayRef input_shape) {
  return impl::spdiags_backward_impl(
      grad_out, offsets, input_shape, _spdiags_backward_kernel_cpu);
}

} // namespace native
} // namespace at
