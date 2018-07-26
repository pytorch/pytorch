#pragma once

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/Error.h"

namespace at {
struct AT_API SparseTensorImpl : public TensorImpl {
  // Stored in COO format, indices + values.

#ifndef USE_TH_SIZE_ZERO_DIM
  // Ideal INVARIANTS:
#else
  // INVARIANTS:
#endif
  // _sparseDims: range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  // _denseDims : range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  // _indices.shape: dimensionality: 2,  shape: (_sparseDims, nnz)
  // _values.shape:  dimensionality: 1 + _denseDims.  shape: (nnz, shape[_sparseDims:])

#ifndef USE_TH_SIZE_ZERO_DIM
  // Actual INVARIANT differences:
  // 1) _sparseDims: range [1, len(shape)] (i.e. we don't allow 0 sparse dimensions)
  // 2) when nnz = 0, there is strange behavior because we lack 0-dimensional sparse tensors.  Namely:
  //    dimensionality == 0, _sparseDims == 0, _denseDims == 0, _indices.shape == {0}, _values.shape == {0}
  // 3) For both _indices.shape and _values.shape, the nnz dimension may be larger than nnz
  // 4) For _values.shape, the non-nnz dimensions may be smaller than the corresponding dimension size, e.g.
  //    a shape (2,3) sparse tensor with _sparseDims == 1, may have _values.shape: (nnz, <=2, <=3).
#endif


  // The true size of the sparse tensor (e.g., if you called to_dense()
  // on it).  When THTensor merges into TensorImpl, this field
  // should move to the parent class.
  std::vector<int64_t> size_;

#ifndef USE_TH_SIZE_ZERO_DIM
  // The number of non-zero elements.
#else
  // The number of non-zero elements, which is guaranteed to match the
  // corresponding nnz dimensions of indices/values.
#endif
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
  explicit SparseTensorImpl(Type * type);

  int64_t nnz() const { return nnz_; }
  int64_t sparseDims() const { return sparseDims_; }
  int64_t denseDims() const { return denseDims_; }
  bool coalesced() const { return coalesced_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }

  IntList sizes() const override;
  IntList strides() const override;
  int64_t dim() const override;
  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override;
  void * unsafeGetTH(bool retain) override;
  std::unique_ptr<Storage> storage() override;

#ifndef USE_TH_SIZE_ZERO_DIM
  // Some ops do some manual size fiddling.
  // TODO: Figure out a more safe way to provide this functionality
  std::vector<int64_t>& _sizes_mut() { return size_; }
#endif

  // WARNING: This function does NOT preserve invariants of sparseDims/denseDims with
  // respect to indices and values
  void raw_resize_(int64_t sparseDims, int64_t denseDims, ArrayRef<int64_t> size) {
#ifndef USE_TH_SIZE_ZERO_DIM
    // UGHHHHH.  Legacy special case
    if (size.size() == 0) {
      size_ = {0};
    } else {
      size_ = size.vec();
    }
#else
    size_ = size;
#endif
    sparseDims_ = sparseDims;
    denseDims_ = denseDims;
  }

#ifdef USE_TH_SIZE_ZERO_DIM
  // NOTE: This function preserves invariants of sparseDims/denseDims with respect to
  // indices and values
  void resize_(int64_t sparseDims, int64_t denseDims, ArrayRef<int64_t> size) {
    AT_CHECK(sparseDims + denseDims == size.size(), "number of dimensions must be sparseDims (", sparseDims, ") + denseDims (", denseDims, "), but got ", size.size());
    AT_CHECK(indices().size(0) == sparseDims, "indices has incorrect first dimension, expected ", sparseDims, ", got ", indices().size(0));
    AT_CHECK(values().dim() == denseDims + 1, "values has incorrect number of dimensions, expected ", denseDims + 1, ", got ", values().dim());

    auto dense_size_original = sizes().slice(sparseDims_);
    auto dense_size_new = size.slice(sparseDims);
    AT_CHECK(
      dense_size_original.equals(dense_size_new),
      "dense dim sizes don't match, expected ", dense_size_original, ", got ", dense_size_new
    );

    size_ = size;
    sparseDims_ = sparseDims;
    denseDims_ = denseDims;
  }

  // NOTE: this function will resize the sparse tensor and also set `indices` and `values` to empty.
  // The reason we do this is that it's difficult to preserve the dim invariants when we change the
  // sparse tensor size but it doesn't agree with the dims of the existing indices or values, so we
  // should also reset the indices and values to preserve the dim invariants.
  void resize_and_clear_(int64_t sparseDims, int64_t denseDims, ArrayRef<int64_t> size) {
    AT_CHECK(sparseDims + denseDims == size.size(), "number of dimensions must be sparseDims (", sparseDims, ") + denseDims (", denseDims, "), but got ", size.size());
    size_ = size;
    sparseDims_ = sparseDims;
    denseDims_ = denseDims;

    auto empty_indices = indices().type().tensor({sparseDims, 0});
    std::vector<int64_t> values_size = {0};
    auto dense_size = sizes().slice(sparseDims);
    values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
    auto empty_values = values().type().tensor(values_size);
    set_indices_and_values_unsafe(empty_indices, empty_values);
  }
#endif

#ifndef USE_TH_SIZE_ZERO_DIM
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
#endif

  void set_coalesced(bool coalesced) { coalesced_ = coalesced; }
#ifndef USE_TH_SIZE_ZERO_DIM
  void set_nnz(int64_t nnz) { nnz_ = nnz; }
#endif

  // Takes indices and values and directly puts them into the sparse tensor, no copy.
  // NOTE: this function is unsafe because it doesn't check whether any indices are
  // out of boundaries of `sizes`, so it should ONLY be used where we know that the
  // indices are guaranteed to be within bounds.
  // This used to be called THSTensor_(_move)
  // NB: This used to be able to avoid a refcount bump, but I was too lazy to
  // make it happen
  void set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values);
};

} // namespace at
