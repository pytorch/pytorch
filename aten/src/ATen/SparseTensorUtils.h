#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/Parallel.h>

namespace at { namespace sparse {

// Just for documentary purposes
using SparseTensor = Tensor;
using SparseType = Type;


// This is an internal utility function for getting at the SparseTensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
inline  SparseTensorImpl* get_sparse_impl(const SparseTensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
inline void alias_into_sparse(const SparseTensor& self, const Tensor& indices, const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the sparse
// indices/values.  This used to be called THSTensor_(_set)
inline void copy_into_sparse(const SparseTensor& self, const Tensor& indices, const Tensor& values, bool non_blocking) {
  alias_into_sparse(
      self,
      indices.to(self._indices().options(), non_blocking, /*copy=*/true),
      values.to(self._values().options(), non_blocking, /*copy=*/true));
}

// TODO: put this into the public API
inline bool is_same_tensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

inline bool is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() && self.dense_dim() == src.dense_dim();
}

// Give us a new values tensor, with the same dimensionality
// as 'values' but with a new number of non-zero elements.
// TODO: Expose this for real in ATen, some day?
// NB: Doesn't preserve data.
inline Tensor new_values_with_size_of(const Tensor& values, int64_t nnz) {
  std::vector<int64_t> size = values.sizes().vec();
  size[0] = nnz;
  return at::empty(size, values.options());
}

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
TORCH_API Tensor flatten_indices(const Tensor& indices, IntArrayRef full_size, bool force_clone = false);

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
TORCH_API Tensor flatten_indices_by_dims(const Tensor& indices, const IntArrayRef& sizes, const IntArrayRef& dims_to_flatten);

// Find the CSR representation for a row `indices` from the COO format
TORCH_API Tensor coo_to_csr(const int64_t* indices, int64_t dim, int64_t nnz);

}} // namespace at::sparse
