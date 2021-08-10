#include <ATen/SparseTensorUtils.h>

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/Parallel.h>

namespace at { namespace sparse {

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
    std::vector<int64_t> indices_mult_cpu_vec;
    indices_mult_cpu_vec.reserve(sparse_dim);
    int64_t mult = 1;
    for (int64_t i = sparse_dim - 1; i >= 0; i--) {
      indices_mult_cpu_vec[i] = mult;
      mult *= full_size[i];
    }
    auto indices_mult_cpu = at::from_blob(
        indices_mult_cpu_vec.data(),
        // NOLINTNEXTLINE(bugprone-argument-comment)
        /*size=*/{sparse_dim, 1},
        indices.options().device(kCPU));
    // NB: must be blocking because this blob may be freed after this closure,
    //     and non_blocking copy will see garbage.
    auto indices_mult = indices_mult_cpu.to(indices.device(), /*non_blocking=*/false);
    // Ideally we want matmul but matmul is slow on CPU Long and not implemented
    // on CUDA Long. So mul is faster.
    return indices.mul(indices_mult).sum(0);
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
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t h, hp0, hp1;
      for (auto i = start; i < end; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) {
          for (h = hp0; h < hp1; h++) {
            csr_accessor[h+1] = i+1;
          }
        }
      }
    });
  }
  return csr;
}

}} // namespace at::sparse
