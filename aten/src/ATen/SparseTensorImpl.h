#pragma once

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/core/Error.h"

namespace at {
struct AT_API SparseTensorImpl : public TensorImpl {
  // Stored in COO format, indices + values.

  // Ideal INVARIANTS:
  // _sparseDims: range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  // _denseDims : range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  // _indices.shape: dimensionality: 2,  shape: (_sparseDims, nnz)
  // _values.shape:  dimensionality: 1 + _denseDims.  shape: (nnz, shape[_sparseDims:])

  // Actual INVARIANT differences:
  // 1) _sparseDims: range [1, len(shape)] (i.e. we don't allow 0 sparse dimensions)
  // 2) when nnz = 0, there is strange behavior because we lack 0-dimensional sparse tensors.  Namely:
  //    dimensionality == 0, _sparseDims == 0, _denseDims == 0, _indices.shape == {0}, _values.shape == {0}
  // 3) For both _indices.shape and _values.shape, the nnz dimension may be larger than nnz
  // 4) For _values.shape, the non-nnz dimensions may be smaller than the corresponding dimension size, e.g.
  //    a shape (2,3) sparse tensor with _sparseDims == 1, may have _values.shape: (nnz, <=2, <=3).


  // The true size of the sparse tensor (e.g., if you called to_dense()
  // on it).  When THTensor merges into TensorImpl, this field
  // should move to the parent class.
  std::vector<int64_t> size_;

  // The number of non-zero elements.
  int64_t nnz_ = 0;

  int64_t sparseDims_ = 0; // number of sparse dimensions
  int64_t denseDims_ = 0; // number of dense dimensions

  Tensor indices_; // always a LongTensor
  Tensor values_;

  // A sparse tensor is 'coalesced' if every index occurs at most once in
  // the indices tensor, and the indices are in sorted order.  (This means
  // that it is very easy to convert a coalesced tensor to CSR format: you
  // need only compute CSR format indices.)
  //
  // Most math operations can only be performed on coalesced sparse tensors,
  // because many algorithms proceed by merging two sorted lists (of indices).
  bool coalesced_ = false;

public:
  // Public for now...
  explicit SparseTensorImpl(at::TensorTypeId, at::ScalarType);

  int64_t nnz() const { return nnz_; }
  int64_t sparseDims() const { return sparseDims_; }
  int64_t denseDims() const { return denseDims_; }
  bool coalesced() const { return coalesced_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }

  IntList sizes() const override;
  IntList strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;

  int64_t dim() const override;
  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override;
  std::unique_ptr<Storage> storage() override;
  at::StorageImpl* storageImpl() const override;
  int64_t storage_offset() const override;

  // Some ops do some manual size fiddling.
  // TODO: Figure out a more safe way to provide this functionality
  std::vector<int64_t>& _sizes_mut() { return size_; }

  // WARNING: This function does NOT preserve invariants of sparseDims/denseDims with
  // respect to indices and values
  void raw_resize_(int64_t sparseDims, int64_t denseDims, ArrayRef<int64_t> size) {
    // UGHHHHH.  Legacy special case
    if (size.size() == 0) {
      size_ = {0};
    } else {
      size_ = size.vec();
    }
    sparseDims_ = sparseDims;
    denseDims_ = denseDims;
  }

  // TODO: I hate these two setters, please get rid of them!!!
  void set_indices(const Tensor& indices) {
    AT_ASSERT(indices.type().backend() == at::toDense(type().backend()));
    AT_ASSERT(indices.type().scalarType() == kLong);
    indices_ = indices;
  }
  void set_values(const Tensor& values) {
    AT_ASSERT(values.type().toSparse() == type());
    values_ = values;
  }

  void set_coalesced(bool coalesced) { coalesced_ = coalesced; }
  void set_nnz(int64_t nnz) { nnz_ = nnz; }

  // This used to be called THSTensor_(_move)
  // NB: This used to be able to avoid a refcount bump, but I was too lazy to
  // make it happen
  void set_indices_and_values(const Tensor& indices, const Tensor& values);
};

} // namespace at
