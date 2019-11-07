#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>

namespace at { namespace sparse {

// Just for documentary purposes
using SparseTensor = Tensor;
using LongTensor = Tensor;
using IntTensor = Tensor;
using SparseType = Type;

// This is an internal utility function for getting at the SparseTensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
inline SparseTensorImpl* get_sparse_impl(const SparseTensor& self) {
  AT_ASSERTM(!self.is_variable(), "_internal_get_SparseTensorImpl: should not be a variable");  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERTM(self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
inline void alias_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the sparse
// indices/values.  This used to be called THSTensor_(_set)
inline void copy_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values, bool non_blocking) {
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

inline Tensor new_values_with_size_of(const Tensor& values, int64_t nnz, ScalarType dtype) {
  std::vector<int64_t> size = values.sizes().vec();
  size[0] = nnz;
  return at::empty(size, values.options().dtype(dtype));
}

inline std::tuple<const Tensor, const Tensor> 
promoted_tensors(const Tensor & first, const Tensor & second, ScalarType commonDtype) {
  const Tensor& first_promoted = first.scalar_type() != commonDtype? first.to(commonDtype) : first;
  const Tensor& second_promoted = second.scalar_type() != commonDtype? second.to(commonDtype) : second;
  return std::tie(first_promoted, second_promoted);
}

// NOTE [ Flatten Sparse Indices ]
// This helper function flattens a sparse indices tensor (a LongTensor) into a 1D
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
inline LongTensor flatten_indices(const Tensor& indices, IntArrayRef full_size, bool force_clone = false) {
  int64_t sparse_dim = indices.size(0);
  if (sparse_dim == 1) {
    if (force_clone) {
      return indices.squeeze(0).clone();
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
inline LongTensor flatten_indices_by_dims(const LongTensor& indices, const IntArrayRef& sizes, const IntArrayRef& dims_to_flatten){
  LongTensor new_indices = at::zeros({indices.size(1)}, indices.options());
  for (auto d : dims_to_flatten) {
    new_indices.mul_(sizes[d]);
    new_indices.add_(indices.select(0, d));
  }
  return new_indices;
}

}} // namespace at::sparse
