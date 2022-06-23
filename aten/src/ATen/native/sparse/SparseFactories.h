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
#include <iostream>
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

  TORCH_CHECK(
      offsets_1d.unsqueeze(0)
          .permute({1, 0})
          .eq(offsets_1d)
          .sum(-1)
          .equal(at::ones_like(offsets_1d)),
      "spdiags(): Offset tensor contains duplicate values");

  const auto n_diag = offsets_1d.size(0);

  auto nnz_per_diag = at::zeros_like(offsets_1d);

  auto setup_iter = TensorIteratorConfig()
                        .add_output(nnz_per_diag)
                        .add_input(offsets_1d)
                        .build();

  // Checks offsets and computes nnz per diag
  setup_func(setup_iter, shape[0], shape[1], diagonals_2d.size(1));

  auto nnz_per_diag_cumsum = nnz_per_diag.cumsum(-1);
  // const auto nnz = nnz_per_diag_cumsum.select(0, -1).item<int64_t>();
  // Note the above fails when no diagonals are provided, this case is allowed
  // by scipy.
  const auto nnz = nnz_per_diag.sum(-1).item<int64_t>();
  auto result_mem_offsets = nnz_per_diag_cumsum.sub_(nnz_per_diag);

  auto indices = at::empty({2, nnz}, offsets_1d.options());
  auto values = at::empty({nnz}, diagonals_2d.options());

  Tensor diag_index = at::arange(n_diag, offsets_1d.options());
  // cpu_kernel requires an output
  auto dummy = at::empty({1}, offsets_1d.options());

  auto iter = TensorIteratorConfig()
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
} // namespace impl
} // namespace native
} // namespace at
