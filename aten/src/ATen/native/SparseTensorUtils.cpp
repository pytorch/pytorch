#include <ATen/native/SparseTensorUtils.h>

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

DEFINE_DISPATCH(flatten_indices_stub);

} // namespace at::native

namespace at::sparse {

// NOTE [ Flatten Sparse Indices ]
// This helper function flattens a sparse indices tensor (a Tensor) into a 1D
// indices tensor. E.g.,
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   full_size = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// In other words, assuming that each `indices[i, :]` is a valid index to a
// tensor `t` of shape `full_size`. This returns the corresponding indices to
// the flattened tensor `t.reshape( prod(full_size[:indices.size(0)]), -1 )`.
// if forceClone is true, the result will forced to be a clone of self.
// if force_clone is true, the result will forced to be a clone of self.
Tensor flatten_indices(const Tensor& indices, IntArrayRef full_size, bool force_clone /*= false*/) {
  int64_t sparse_dim = indices.size(0);
  if (sparse_dim == 1) {
    if (force_clone) {
      return indices.squeeze(0).clone(at::MemoryFormat::Contiguous);
    } else {
      return indices.squeeze(0);
    }
  } else {
    if (!indices.numel()) {
      return at::zeros({indices.size(1)}, indices.options().dtype(kLong));
    }
    return at::native::flatten_indices_stub(indices.device().type(), indices, full_size.slice(0, sparse_dim));
  }
}

// Flatten sparse tensor's indices from nD to 1D, similar to NOTE [ Flatten Sparse Indices ],
// except this one allows partial flatten: only flatten on specified dims. Note that
// the flatten indices might be uncoalesced if dims_to_flatten.size() < sparse_dim.
// Also if input indices is already coalesced, the flattened indices will also be sorted.
//
// args:
//    indices: sparse tensor indices
//    sizes: sparse tensor sizes
//    dims_to_flatten: a list of dim index to flatten
//
// Ex1:
//   indices = [[2, 4, 0],
//             [3, 1, 3]]
//   sizes = [2, 12]
//   dims_to_flatten = [0, 1]
//   new_indices = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 3 ] = [27, 49, 3]
//
// Ex2:
//   dims_to_flatten = [1]
//   new_indices = [ 3, 1, 3 ]  # uncoalesced
Tensor flatten_indices_by_dims(const Tensor& indices, const IntArrayRef& sizes, const IntArrayRef& dims_to_flatten){
  Tensor new_indices = at::zeros({indices.size(1)}, indices.options());
  for (auto d : dims_to_flatten) {
    new_indices.mul_(sizes[d]);
    new_indices.add_(indices.select(0, d));
  }
  return new_indices;
}

Tensor coo_to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
  /*
    Find the CSR representation for a row `indices` from the COO format
    Inputs:
      `indices` is the row pointer from COO indices
      `dim` is the row dimensionality
      `nnz` is the number of non-zeros

    Output:
      `csr` is a compressed row array in a CSR format
  */
  Tensor csr = at::zeros({dim + 1}, kLong);

  // TODO: eliminate this conditional when zero-size dims supported correctly
  if (nnz > 0) {
    auto csr_accessor = csr.accessor<int64_t, 1>();
    // Convert the sparse matrix to CSR format
    at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        auto hp0 = indices[i];
        auto hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) {
          for (int64_t h = hp0; h < hp1; h++) {
            csr_accessor[h+1] = i+1;
          }
        }
      }
    });
  }
  return csr;
}

Tensor zeros_like_with_indices(const Tensor& t) {
  TORCH_INTERNAL_ASSERT(t.is_sparse());
  return at::_sparse_coo_tensor_with_dims_and_tensors(
      t.sparse_dim(),
      t.dense_dim(),
      t.sizes(),
      t._indices().clone(),
      at::zeros({1}, t._values().options()).expand_as(t._values()),
      t.options(),
      t.is_coalesced());
}

Tensor full_coo_indices(IntArrayRef sizes, TensorOptions options) {
  const auto max_size = *std::max_element(sizes.begin(), sizes.end());
  const auto max_size_arange = at::arange(max_size, options);
  std::vector<Tensor> stack;
  stack.reserve(sizes.size());
  for (size_t i=0; i < sizes.size(); i++) {
    Tensor a = max_size_arange.narrow(-1, 0, sizes[i]);
    for (size_t j=0; j < sizes.size(); j++) {
      if (i != j) {
        a.unsqueeze_(j);
      }
    }
    stack.push_back(a.expand(sizes));
  }
  return at::stack(stack).flatten(1, -1);
}

} // namespace at::sparse
