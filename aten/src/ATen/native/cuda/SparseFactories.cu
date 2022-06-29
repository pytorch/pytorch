#include <ATen/Dispatch.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sparse/SparseFactories.h>
#include <c10/util/ArrayRef.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sparse_coo_tensor.h>
#endif

namespace at {
namespace native {

namespace {
void _spdiags_kernel_cuda(
    TensorIterator& iter,
    const Tensor& diagonals,
    Tensor& values,
    Tensor& indices) {
  int64_t* row_index_write_ptr = indices[0].data_ptr<int64_t>();
  int64_t* col_index_write_ptr = indices[1].data_ptr<int64_t>();
  const int64_t diagonals_row_stride = diagonals.stride(0);
  const int64_t diagonals_read_stride = diagonals.stride(1);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      diagonals.scalar_type(),
      "spdiags_cuda",
      [&] {
        scalar_t* values_write_ptr = values.data_ptr<scalar_t>();
        scalar_t* diagonals_ptr = diagonals.data_ptr<scalar_t>();
        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(
                int64_t diag_index,
                int64_t diag_offset,
                int64_t out_offset,
                int64_t n_out) -> int64_t {
              int64_t* __restrict__ rows_start =
                  row_index_write_ptr + out_offset;
              int64_t* __restrict__ cols_start =
                  col_index_write_ptr + out_offset;
              scalar_t* __restrict__ vals_start = values_write_ptr + out_offset;
              int64_t const first_col = ::max(diag_offset, int64_t{0});
              int64_t const first_row = first_col - diag_offset;
              scalar_t* __restrict__ data_read = diagonals_ptr +
                  (diag_index * diagonals_row_stride) +
                  (first_col * diagonals_read_stride);
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

void _spdiags_backward_kernel_cuda(TensorIterator& iter, Tensor& grad_in) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      iter.dtype(),
      "spdiags_backward_cpu",
      [&] {
        auto* grad_in_ptr = grad_in.data_ptr<scalar_t>();
        int64_t const grad_in_row_stride = grad_in.stride(0);
        int64_t const grad_in_col_stride = grad_in.stride(1);
        gpu_kernel(
            iter,
            [grad_in_ptr, grad_in_row_stride, grad_in_col_stride] GPU_LAMBDA(
                scalar_t grad_out_value,
                int64_t row_idx,
                int64_t col_idx) -> scalar_t {
              scalar_t* grad_in_val_ptr = grad_in_ptr +
                  (row_idx * grad_in_row_stride) +
                  (col_idx * grad_in_col_stride);
              *grad_in_val_ptr = grad_out_value;
              return scalar_t{0};
            });
      });
}
} // namespace
REGISTER_DISPATCH(spdiags_kernel_stub, &_spdiags_kernel_cuda)

REGISTER_DISPATCH(spdiags_backward_kernel_stub, &_spdiags_backward_kernel_cuda)

} // namespace native
} // namespace at
