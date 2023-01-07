#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/sparse/SparseFactories.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/where.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(spdiags_kernel_stub);

Tensor spdiags(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout) {
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
      "Offset Tensor must have dtype Long but got ",
      offsets_1d.scalar_type());

  TORCH_CHECK(
      offsets_1d.numel() == std::get<0>(at::_unique(offsets_1d)).numel(),
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
  spdiags_kernel_stub(iter.device_type(), iter, diagonals_2d, values, indices);
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

} // namespace native
} // namespace at
