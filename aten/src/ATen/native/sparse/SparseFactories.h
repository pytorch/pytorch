#pragma once
#include <ATen/SparseTensorUtils.h>
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

template <typename kernel_func_t>
Tensor spdiags_impl(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout,
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
        "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, got ",
        *layout);
  }
  TORCH_CHECK(
      offsets_1d.scalar_type() == at::kLong,
      "Offset Tensor must has dtype Long but got ",
      offsets_1d.scalar_type());

  TORCH_CHECK(
      offsets_1d.unsqueeze(1)
          .eq(offsets_1d)
          .sum(-1)
          .equal(at::ones_like(offsets_1d)),
      "Offset tensor contains duplicate values");

  auto nnz_per_diag = at::where(
      offsets_1d.le(0),
      offsets_1d.add(shape[0]).clamp_max_(diagonals_2d.size(1)),
      offsets_1d.add(-std::min<int64_t>(shape[1], diagonals_2d.size(1))).neg());

  auto nnz_per_diag_cumsum = nnz_per_diag.cumsum(-1);
  const auto nnz = diagonals_2d.size(0) > 0
      ? nnz_per_diag_cumsum.select(-1, -1).item<int64_t>()
      : int64_t{0};
  // Offsets into nnz for each diagonal
  auto result_mem_offsets = nnz_per_diag_cumsum.sub(nnz_per_diag);
  // coo tensor guts
  auto indices = at::empty({2, nnz}, offsets_1d.options());
  auto values = at::empty({nnz}, diagonals_2d.options());
  // We add this indexer to lookup the row of diagonals we are reading from at
  // each iteration
  const auto n_diag = offsets_1d.size(0);
  Tensor diag_index = at::arange(n_diag, offsets_1d.options());
  // cpu_kernel requires an output
  auto dummy = at::empty({1}, offsets_1d.options()).resize_({0});
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .add_output(dummy)
                  .add_input(diag_index)
                  .add_input(offsets_1d)
                  .add_input(result_mem_offsets)
                  .add_input(nnz_per_diag)
                  .build();
  kernel_func(iter, diagonals_2d, values, indices);
  auto result_coo = at::sparse_coo_tensor(indices, values, shape);
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

template <typename kernel_func_t>
Tensor spdiags_backward_impl(
    const Tensor& grad_out,
    const Tensor& offsets,
    IntArrayRef input_shape,
    const kernel_func_t& kernel_func) {
  auto offsets_1d = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;

  auto n_diag = input_shape.size() == 2 ? input_shape[0] : 1;
  auto n_col_in = input_shape.size() == 2 ? input_shape[1] : input_shape[0];
  AT_ASSERT(grad_out.dim() == 2);
  AT_ASSERT(offsets_1d.size(0) == n_diag);
  auto grad_in_options = grad_out.options().layout(Layout::Strided);
  // zeros since we are only going to se the non-zero elements
  Tensor grad_in = at::zeros({n_diag, n_col_in}, grad_in_options);
  auto grad_out_coo =
      grad_out.layout() == Layout::Sparse ? grad_out : grad_out.to_sparse();

  auto grad_out_values = sparse::get_sparse_impl(grad_out_coo)->values_;
  auto grad_out_indices = sparse::get_sparse_impl(grad_out_coo)->indices_;
  auto grad_out_row_indices = grad_out_indices[0];
  auto grad_out_col_indices = grad_out_indices[1];

  // Precompute input row indices for each nnz
  auto row_in_indices =
      offsets_1d
          .eq(grad_out_indices[1].sub(grad_out_indices[0]).reshape({-1, 1}))
          .nonzero()
          .permute({1, 0})[-1];

  // dummy output required by cpu/gpu_kernel
  auto dummy = at::empty({1}, grad_in.options());
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(dummy)
                  .add_input(grad_out_values)
                  .add_input(row_in_indices)
                  .add_input(grad_out_col_indices)
                  .build();

  kernel_func(iter, grad_in);

  return grad_in.reshape(input_shape);
}
} // namespace impl
} // namespace native
} // namespace at
