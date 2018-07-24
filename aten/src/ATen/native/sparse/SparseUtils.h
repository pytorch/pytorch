#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>

#include <TH/THGeneral.h>

namespace at { namespace native {

// Just for documentary purposes
using SparseTensor = Tensor;
using LongTensor = Tensor;
using IntTensor = Tensor;
using SparseType = Type;

namespace {

// This is an internal utility function for getting at the SparseTensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
SparseTensorImpl* _get_sparse_impl(const SparseTensor& self) {
  if (!self.is_sparse()) AT_ERROR("_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Port of the old THCSTensor_(checkGPU), but it doesn't really belong here
// because it is more general
// NB: I dropped kernelP2PEnabled support
// NB: This only works if the tensors are KNOWN to be CUDA.
// TODO: Generalize it so it works on CPU as well
inline bool _check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  const Tensor& ref_t = ts.front();
  int64_t curDevice = current_device();
  for (const Tensor& t : ts) {
    if (t.get_device() != curDevice) return false;
  }
  return true;
}

inline void _raw_resize_sparse(const SparseTensor& self, int64_t sparseDims, int64_t denseDims, IntList size) {
  _get_sparse_impl(self)->raw_resize_(sparseDims, denseDims, size);
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
inline void _alias_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
  _get_sparse_impl(self)->set_indices_and_values(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the sparse
// indices/values.  This used to be called THSTensor_(_set)
inline void _copy_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
  _alias_into_sparse(self, indices.clone(), values.clone());
}

// Does NOT make copies of indices/values
inline SparseTensor _new_with_dims_and_tensor_sparse(
    const SparseType& dtype,
    int64_t sparseDims,
    int64_t denseDims,
    ArrayRef<int64_t> sizes,
    const LongTensor& indices,
    const Tensor& values) {
  SparseTensor self = new_sparse(dtype);
  _raw_resize_sparse(self, sparseDims, denseDims, sizes);
  _alias_into_sparse(self, indices, values);
  return self;
}

// TODO: put this into the public API
inline bool isSameTensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

inline bool _is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self._sparseDims() == src._sparseDims() && self._denseDims() == src._denseDims();
}

// if forceClone is true, the result will forced to be a clone of self.
inline LongTensor _newFlattenedIndices(const SparseTensor& self, bool forceClone) {
  LongTensor indices = self._indices();
  int64_t sparseDims = self._sparseDims();
  if (sparseDims == 1) {
    if (forceClone) {
      return indices.clone();
    } else {
      return indices;
    }
  } else {
    // FIXME TH_INDEX_BASE
    int64_t factor = 1;
    LongTensor indices1D = at::empty({1, self._nnz()}, indices.options());
    indices1D.fill_(TH_INDEX_BASE);
    for (int64_t d = sparseDims - 1; d >= 0; d--) {
      indices1D.add_(indices.select(0, d), factor);
      if (TH_INDEX_BASE != 0) {
        indices1D.add_(-TH_INDEX_BASE);
      }
      factor *= self.size(d);
    }
    return indices1D;
  }
}

// Give us a new values tensor, with the same dimensionality
// as 'values' but with a new number of non-zero elements.
// TODO: Expose this for real in ATen, some day?
// NB: Doesn't preserve data.
inline Tensor _new_values_with_size_of(const Tensor& values, int64_t nnz) {
  if (values.numel() == 0) { // values tensor uninitialized
    // TODO: This logic looks bogus; if we have an uninitialized
    // values tensor, why should we believe that denseDims == 0?
    // That's the assumption this code makes.
    return values.type().tensor({nnz});
  } else {
    std::vector<int64_t> size = values.sizes();
    size[0] = nnz;
    return values.type().tensor(size);
  }
}



} // anonymous namespace

}} // namespace at::native
