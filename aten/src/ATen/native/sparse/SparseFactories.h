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

  // We are going to introduce some temporaries to compute nnz_per_diag and do
  // the final check at onece, we induce a block scope to allow these
  // intermediates to be cleaned up before we continue
  auto nnz_per_diag = at::empty_like(offsets_1d);
  {
    auto n_col_out = at::scalar_tensor(shape[1], offsets_1d.options());
    auto n_row_out = at::scalar_tensor(shape[0], offsets_1d.options());
    auto out_of_bounds_mask =
        offsets_1d.le(n_row_out.neg()).logical_and_(offsets_1d.ge(n_col_out));

    TORCH_CHECK(
        out_of_bounds_mask.logical_not().all().item<bool>(),
        "spdiags(): Detected the folowing diagonal offsets which are not supported by the output shape, ",
        offsets.masked_select(out_of_bounds_mask),
        ". Valid offsets are in the range [",
        (-shape[0]) + 1,
        ",",
        shape[1] + 1,
        ")");
    auto min_cols = at::minimum(
        n_col_out,
        at::scalar_tensor(diagonals_2d.size(1), offsets_1d.options()));
    auto zero = at::scalar_tensor(int64_t{0}, offsets_1d.options());
    // Mask for offsets which are positive
    auto offset_pos_mask = offsets_1d.ge(zero);
    // Seed nnz per diag with the formula for negative offsets
    nnz_per_diag =
        at::minimum_out(nnz_per_diag, n_row_out.add(offsets_1d), min_cols);
    // Compute the sizes for positive offsets
    auto nnz_per_pos_offset_diag =
        at::minimum(min_cols.sub(offsets_1d), n_row_out)
            .masked_select(offset_pos_mask);
    nnz_per_diag.index_put_({offset_pos_mask}, nnz_per_pos_offset_diag);
    // Any negative offsets are clamped to zero
    nnz_per_diag.clamp_min_(zero);
  }
  // auto nnz_per_diag_cumsum = nnz_per_diag.cumsum(-1);
  //  const auto nnz = nnz_per_diag_cumsum.select(0, -1).item<int64_t>();
  //  Note the above fails when no diagonals are provided, this case is allowed
  //  by scipy.
  const auto nnz = nnz_per_diag.sum(-1).item<int64_t>();
  // Offsets into nnz for each diagonal
  auto result_mem_offsets = nnz_per_diag.cumsum(-1).sub_(nnz_per_diag);
  // coo tensor guts
  auto indices = at::empty({2, nnz}, offsets_1d.options());
  auto values = at::empty({nnz}, diagonals_2d.options());
  // We add this indexer to lookup the row of diagonals we are reading from at
  // each iteration
  const auto n_diag = offsets_1d.size(0);
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
