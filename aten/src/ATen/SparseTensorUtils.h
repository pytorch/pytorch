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
  AT_ASSERTM(!self.is_variable(), "_internal_get_SparseTensorImpl: should not be a variable");
  AT_ASSERTM(self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Port of the old THCSTensor_(checkGPU), but it doesn't really belong here
// because it is more general
// NB: I dropped kernelP2PEnabled support
// NB: This only works if the tensors are KNOWN to be CUDA.
// TODO: Generalize it so it works on CPU as well
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  int64_t curDevice = current_device();
  for (const Tensor& t : ts) {
    if (t.get_device() != curDevice) return false;
  }
  return true;
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
inline void alias_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the sparse
// indices/values.  This used to be called THSTensor_(_set)
inline void copy_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values, bool non_blocking) {
  alias_into_sparse(self, self._indices().type().copy(indices, non_blocking), self._values().type().copy(values, non_blocking));
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
inline LongTensor flatten_indices(const Tensor& indices, IntList full_size, bool force_clone = false) {
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
    auto indices_mult_cpu = indices.type().cpu()
                                   .tensorFromBlob(indices_mult_cpu_vec.data(), /*size=*/{sparse_dim, 1});
    // NB: must be blocking because this blob may be freed after this closure,
    //     and non_blocking copy will see garbage.
    auto indices_mult = indices_mult_cpu.to(indices.device(), /*non_blocking=*/false);
    // Ideally we want matmul but matmul is slow on CPU Long and not implemented
    // on CUDA Long. So mul is faster.
    return indices.mul(indices_mult).sum(0);
  }
}

}} // namespace at::sparse
