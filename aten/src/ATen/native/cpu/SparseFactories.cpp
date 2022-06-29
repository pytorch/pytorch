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
              if (n_out > 0) {
                auto* rows_start = row_index_write_ptr + out_offset;
                auto* cols_start = col_index_write_ptr + out_offset;
                auto* vals_start = values_write_ptr + out_offset;
                const int64_t first_col = std::max<int64_t>(diag_offset, 0);
                const int64_t first_row = first_col - diag_offset;
                auto* data_read = diagonals[diag_index].data_ptr<scalar_t>() +
                    first_col * diagonals_read_stride;
                for (int64_t i = 0; i < n_out; ++i) {
                  rows_start[i] = first_row + i;
                  cols_start[i] = first_col + i;
                  vals_start[i] = data_read[i * diagonals_read_stride];
                }
              }
              // dummy return
              return 0;
            });
      });
}

void _spdiags_backward_kernel_cpu(TensorIterator& iter, Tensor& grad_in) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      iter.dtype(),
      "spdiags_backward_cpu",
      [&] {
        auto grad_in_accessor = grad_in.accessor<scalar_t, 2>();
        cpu_kernel(
            iter,
            [&](scalar_t grad_out_value,
                int64_t row_idx,
                int64_t col_idx) -> scalar_t {
              grad_in_accessor[row_idx][col_idx] = grad_out_value;
              return scalar_t{0};
            });
      });
}
} // namespace

REGISTER_DISPATCH(spdiags_kernel_stub, &_spdiags_kernel_cpu)
REGISTER_DISPATCH(spdiags_backward_kernel_stub, &_spdiags_backward_kernel_cpu)

} // namespace native
} // namespace at
