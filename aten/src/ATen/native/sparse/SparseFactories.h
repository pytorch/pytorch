#pragma once
#include <ATen/TensorIndexing.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
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
namespace impl {

template <typename setup_func_t, typename kernel_func_t>
Tensor spdiags_impl(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout,
    const setup_func_t& setup_func,
    const kernel_func_t& kernel_func) {
  auto diagonals_2d = diagonals.dim() == 1 ? diagonals.unsqueeze(0) : diagonals;
  TORCH_CHECK(diagonals_2d.dim() == 2, "Diagonals must be vector or matrix");
  TORCH_CHECK(shape.size() == 2, "Output shape must be 2d");
  auto offsets_1d = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;
  TORCH_CHECK(offsets_1d.dim() == 1, "Offsets must be scalar or vector");
  TORCH_CHECK(
      diagonals_2d.size(0) == offsets_1d.size(0),
      "Number of diagonals (",
      diagonals_2d.size(0),
      ") does not match the number of offsets (",
      offsets_1d.size(0),
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
  TORCH_CHECK(
      offsets_1d.scalar_type() == at::kLong,
      "spdiags(): Expected a LongTensor of offsets but got ",
      offsets_1d.scalar_type());

  auto nnz_per_diag = at::zeros_like(offsets_1d);

  auto setup_iter = TensorIteratorConfig()
                        .add_output(nnz_per_diag)
                        .add_input(offsets_1d)
                        .build();
  int64_t nnz = 0;
  // Checks offsets and computes nnz per diag
  setup_func(setup_iter, shape[0], shape[1], diagonals_2d.size(1), nnz);
  auto n_diag = offsets_1d.size(0);
  auto nnz_prefix = nnz_per_diag.cumsum(0).sub(nnz_per_diag);

  // We need to restride outputs so that the nnz dimension appears to be n_diag,
  // and we will advance it manually
  auto indices = at::zeros({2, nnz}, offsets_1d.options());
  auto row_indices_restrided = indices[0].as_strided({n_diag}, {0});
  auto col_indices_restrided = indices[1].as_strided({n_diag}, {0});
  auto values = at::zeros({nnz}, diagonals_2d.options());
  auto values_restrided = values.as_strided({n_diag}, {0});

  // Diagonals is going to advance normally, but only over rows, we must drop
  // the col dim, so the shapes match
  auto diagonals_restrided =
      diagonals_2d.as_strided({n_diag}, {diagonals_2d.stride(0)});

  // We build an index value buffer so we can do fast copies into the indices
  // view
  auto max_idx = std::max(shape[0], shape[1]);
  Tensor idx_buffer = at::arange(max_idx, offsets_1d.options());
  auto idx_buffer_restrided = idx_buffer.as_strided({n_diag}, {0});

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(values_restrided)
                  .add_output(row_indices_restrided)
                  .add_output(col_indices_restrided)
                  .add_input(diagonals_restrided)
                  .add_input(offsets_1d)
                  .add_input(idx_buffer_restrided)
                  .add_input(nnz_per_diag)
                  .add_input(nnz_prefix)
                  .build();

  kernel_func(iter, diagonals_2d.stride(1));
  auto result_coo = at::sparse_coo_tensor(indices, values, shape).coalesce();
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
} // namespace impl
} // namespace native
} // namespace at
