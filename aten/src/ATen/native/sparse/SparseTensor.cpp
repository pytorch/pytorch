// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/BlasUtils.h>

namespace at { namespace native {

// Just for documentary purposes
using SparseTensor = Tensor;
using LongTensor = Tensor;
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
}

/******************************************************************************
 * access methods
 ******************************************************************************/

int64_t _dimI_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->dimI();
}

int64_t _dimV_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->dimV();
}

bool is_coalesced_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->coalesced();
}

int64_t _nnz_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->nnz();
}

Tensor _indices_sparse(const SparseTensor& self) {
  auto nnz = self._nnz();
  if (nnz == 0) {
    // Narrows don't work on 0-length tensors
    // TODO: When we handle zero-size dims correctly, this will work and
    // we can remove the special case.
    return _get_sparse_impl(self)->indices();
  }
  return _get_sparse_impl(self)->indices().narrow(1, 0, nnz);
}

Tensor _values_sparse(const SparseTensor& self) {
  // See indices for some relevant notes
  auto nnz = self._nnz();
  if (nnz == 0) {
    return _get_sparse_impl(self)->values();
  }
  return _get_sparse_impl(self)->values().narrow(0, 0, nnz);
}

/******************************************************************************
 * creation methods
 ******************************************************************************/

/* Empty init */
SparseTensor new_sparse(const SparseType& dtype) {
  AT_ASSERT(!dtype.is_variable_or_undefined());
  AT_ASSERT(dtype.is_sparse());
  // TODO: Hmm... this const_cast business seems a bit dodgy
  return SparseTensor(new SparseTensorImpl(const_cast<SparseType*>(&dtype)), /* retain */ false);
}

/*** Helper methods ***/

namespace {
  void _raw_resize_sparse(const SparseTensor& self, int64_t dimI, int64_t dimV, ArrayRef<int64_t> size) {
    _get_sparse_impl(self)->raw_resize_(dimI, dimV, size);
  }

  // Takes indices and values and directly puts them into the sparse tensor, no
  // copy.  This used to be called THSTensor_(_move)
  void _alias_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
    _get_sparse_impl(self)->set_indices_and_values(indices, values);
  }

  // Take indices and values and makes a (data) copy of them to put into the sparse
  // indices/values.  This used to be called THSTensor_(_set)
  void _copy_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
    _alias_into_sparse(self, indices.clone(), values.clone());
  }

  // Does NOT make copies of indices/values
  SparseTensor _new_with_dims_and_tensor_sparse(
      const SparseType& dtype,
      int64_t dimI,
      int64_t dimV,
      ArrayRef<int64_t> sizes,
      const LongTensor& indices,
      const Tensor& values) {
    SparseTensor self = new_sparse(dtype);
    _raw_resize_sparse(self, dimI, dimV, sizes);
    _alias_into_sparse(self, indices, values);
    return self;
  }
}

/* Pointer-copy init */
SparseTensor new_with_tensor_sparse(const LongTensor& indices, const Tensor& values) {
  const SparseType& dtype = values.type().toSparse();

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t dimI = indices.size(0);
  int64_t dimV = values.dim() - 1;

  std::vector<int64_t> computed_sizes(dimI + dimV);
  LongTensor computed_indices_sizes = std::get</* indices */ 1>(indices.max(/* dim */ 1, /* keepdim */ true));
  computed_indices_sizes.add_(1); // len = max_index + 1
  auto computed_indices_sizes_accessor = computed_indices_sizes.accessor<int64_t, 1>();
  for (int64_t d = 0; d < dimI; d++) {
    computed_sizes[static_cast<size_t>(d)] = computed_indices_sizes_accessor[d];
  }
  for (int64_t d = 0; d < dimV; d++) {
    computed_sizes[static_cast<size_t>(dimI + d)] = values.size(d+1);
  }
  return _new_with_dims_and_tensor_sparse(dtype, dimI, dimV, computed_sizes, indices, values);
}

SparseTensor new_with_size_sparse(const SparseType& dtype, ArrayRef<int64_t> size) {
  SparseTensor self = new_sparse(dtype);
  _raw_resize_sparse(self, size.size(), 0, size);
  return self;
}

// NB: Got rid of the sizes == NULL case
SparseTensor new_with_tensor_and_size_unsafe_sparse(const LongTensor& indices, const Tensor& values, ArrayRef<int64_t> sizes) {
  const SparseType& dtype = values.type().toSparse();
  if (indices.dim() == 0 && values.dim() == 0) {
    return new_with_size_sparse(dtype, sizes);
  }

  int64_t dimI = indices.size(0);
  int64_t dimV = values.dim() - 1;
  return _new_with_dims_and_tensor_sparse(dtype, dimI, dimV, sizes, indices, values);
}

// NB: Got rid of the sizes == NULL case
SparseTensor new_with_tensor_and_size_sparse(const LongTensor& indices, const Tensor& values, ArrayRef<int64_t> sizes) {
  const SparseType& dtype = values.type().toSparse();
  if (indices.dim() == 0 && values.dim() == 0) {
    return new_with_size_sparse(dtype, sizes);
  }

  int64_t dimI = indices.size(0);
  int64_t dimV = values.dim() - 1;
  AT_CHECK(sizes.size() == dimI + dimV, "number of dimensions must be dimI + dimV");

  LongTensor max_indices = std::get</* indices */ 1>(indices.max(/* dim */ 1, /* keepdim */ false));
  auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
  for (int64_t d = 0; d < dimI; d++) {
    int64_t max_index_in_dim = max_indices_accessor[d];
    int64_t dim_size = sizes[static_cast<size_t>(d)];
    AT_CHECK(max_index_in_dim < dim_size,
             "sizes is inconsistent with indices: for dim ", d, ", size is ", dim_size, " but found index ", max_index_in_dim);
  }
  for (int64_t d = 0; d < dimV; d++) {
    int64_t values_size = values.size(d+1);
    int64_t specified_size = sizes[static_cast<size_t>(dimI + d)];
    AT_CHECK(values_size < specified_size,
             "values and sizes are inconsistent: sizes[", d + dimI, "] is ", specified_size,
             " but values.size(", d + 1, ") is ", values_size);
  }
  return _new_with_dims_and_tensor_sparse(dtype, dimI, dimV, sizes, indices, values);
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(const SparseTensor& self) {
  SparseTensor other = new_sparse(self.type());
  _raw_resize_sparse(other, self._dimI(), self._dimV(), self.sizes());
  _copy_into_sparse(other, self._indices(), self._values());
  _get_sparse_impl(other)->set_coalesced(self.is_coalesced());
  _get_sparse_impl(other)->set_nnz(self._nnz());
  return other;
}

SparseTensor transpose_sparse(const SparseTensor& self, int64_t d1, int64_t d2) {
  SparseTensor other = clone_sparse(self);
  other.transpose_(d1, d2);
  return other;
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

/*
// We should implement a utility function which: (1) sets nnz and (2) resizes
// indices/values to hold enough space to fit nnz, if nnz is larger than
// the previous amount.  This ensures that we maintain the nnz invariant.
void _resize_nnz_(const SparseTensor& self, int64_t nnz) {
}
*/

void resize_sparse(const SparseTensor& self, ArrayRef<int64_t> size) {
  _raw_resize_sparse(self, size.size(), 0, size);
}

SparseTensor& raw_resize_sparse_(SparseTensor& self, ArrayRef<int64_t> size, int64_t dimI, int64_t dimV) {
  if (dimI == -1) {
    dimI = self._indices().size(0);
  }
  if (dimV == -1) {
    dimV = self._values().dim() - 1;
  }
  _raw_resize_sparse(self, dimI, dimV, size);
  return self;
}

namespace {
  bool _is_same_size_as_sparse(const SparseTensor& self, const SparseTensor& src) {
    return self._dimI() == src._dimI() && self._dimV() == src._dimV() && self.sizes().equals(src.sizes());
  }
}

SparseTensor& resize_as_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    _raw_resize_sparse(self, src._dimI(), src._dimV(), src.sizes());
  }
  return self;
}

// NB: Dropped the resizeNd variants

Tensor sparse_to_dense(const SparseTensor& self) {
  Tensor dst = self.type().toDense().zeros(self.sizes());
  return dst.add_(self);
}

void copy_sparse(const SparseTensor& self, const SparseTensor& src) {
  if (self.equal(src)) return;
  _raw_resize_sparse(self, src._dimI(), src._dimV(), src.sizes());
  _copy_into_sparse(self, src._indices(), src._values());
  _get_sparse_impl(self)->set_coalesced(src.is_coalesced());
  _get_sparse_impl(self)->set_nnz(src._nnz());
}

void transpose_sparse_(const SparseTensor& self, int64_t d1, int64_t d2) {
  int64_t dimI = self._dimI();
  AT_CHECK(d1 < dimI && d2 < dimI, "Transposed dimensions should be sparse.  Got dimI: ", dimI, ", d1: ", d1, ", d2: ", d2);
  LongTensor indices = self._indices();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto nnz = self._nnz();
  for (int64_t i = 0; i < nnz; i++) {
    int64_t tmp = indices_accessor[d1][i];
    indices_accessor[d1][i] = indices_accessor[d2][i];
    indices_accessor[d2][i] = tmp;
  }
  auto& sizes = _get_sparse_impl(self)->_sizes_mut(); // TODO: Do this more safely
  std::swap(sizes[d1], sizes[d2]);
}

SparseTensor coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(!self.is_variable_or_undefined());
  AT_ASSERT(self.is_sparse());

  if (self._nnz() < 2) {
    _get_sparse_impl(self)->set_coalesced(true);
  }
  if (self.is_coalesced()) {
    return self;
  }

  LongTensor indices = self._indices();
  Tensor values = self._values().contiguous();
  int64_t dimI = self._dimI();
  int64_t dimV = self._dimV();
  int64_t nnz = self._nnz();

  LongTensor indices_scalar = at::CPU(kLong).zeros({nnz});

  int64_t factor = 1;
  for (int64_t d = dimI - 1; d >= 0; d--) {
    LongTensor indices_slice = indices.select(0, d);
    indices_scalar.add_(indices_slice, factor); // cadd is swapped args
    factor *= self.size(d);
  }

  SparseTensor dst = new_sparse(values.type().toSparse());
  _raw_resize_sparse(dst, dimI, dimV, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  LongTensor newIndices = indices.type().tensor(indices.sizes());
  Tensor newValues = values.type().tensor(values.sizes());
  _alias_into_sparse(dst, newIndices, newValues);

  LongTensor indicesBuffer;
  LongTensor indicesPermutation;
  std::tie(indicesBuffer, indicesPermutation) = indices_scalar.sort(0);
  auto newIndicesAccessor = newIndices.accessor<int64_t, 2>();
  auto indicesAccessor = indices.accessor<int64_t, 2>();
  auto indicesPermutationAccessor = indicesPermutation.accessor<int64_t, 1>();
  auto indicesBufferAccessor = indicesBuffer.accessor<int64_t, 1>();

  int64_t i = -1;
  AT_DISPATCH_ALL_TYPES(
      values.type(), "coalesce", [&] {
        int64_t prev = -1;
        int64_t blockSize = values.stride(0);
        scalar_t* values_ptr = values.data<scalar_t>();
        scalar_t* newValues_ptr = newValues.data<scalar_t>();
        for (int64_t j = 0; j < nnz; j++) {
          int64_t pos = indicesPermutationAccessor[j];
          int64_t curr = indicesBufferAccessor[j];
          if (curr == prev) {
            thblas::axpy<scalar_t>(blockSize, 1, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
          } else {
            ++i;
            for (int64_t d = 0; d < dimI; d++) {
              newIndicesAccessor[d][i] = indicesAccessor[d][pos];
            }
            thblas::copy<scalar_t>(blockSize, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
          }
          prev = curr;
        }
    });

  _get_sparse_impl(dst)->set_coalesced(true);
  _get_sparse_impl(dst)->set_nnz(i + 1);

  return dst;
}

void sparse_mask_out_cpu(SparseTensor& r, const Tensor& t, const SparseTensor& mask) {
  AT_CHECK(mask.is_coalesced(), "mask is uncoalesced");
  resize_as_sparse_(r, mask);
  if (mask._nnz() == 0) {
    r.zero_();
    return;
  }
  int64_t dim = t.dim();
  int64_t dimI = mask._dimI();
  LongTensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = t._values().type().tensor(mask_values.sizes());
  _alias_into_sparse(r, mask_indices.clone(), r_values);
  _get_sparse_impl(r)->set_coalesced(mask.is_coalesced());
  int64_t r_nnz = mask._nnz();
  _get_sparse_impl(r)->set_nnz(r_nnz);
  auto mask_indices_accessor = mask_indices.accessor<int64_t, 2>();

  if (dim > dimI) {
    // NB: This used to reuse buffers, but I deoptimized it
    for (int64_t i = 0; i < r_nnz; i++) {
      Tensor srcBuffer = t;
      for (int64_t d = 0; d < dimI; d++) {
        srcBuffer = srcBuffer.select(0, mask_indices_accessor[d][i]);
      }
      Tensor dstBuffer = r_values.select(0, i);
      dstBuffer.copy_(srcBuffer);
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        r_values.type(), "sparse_mask", [&] {
          auto r_values_accessor = r_values.accessor<scalar_t, 1>();
          // NB: The old code did this pointer access in a weird way (going straight
          // to storage + storageOffset.)  Was there perhaps a method to the
          // madness?
          scalar_t* t_ptr = t.data<scalar_t>();
          for (int64_t i = 0; i < r_nnz; i++) {
            int64_t idx = 0;
            for (int64_t d = 0; d < dimI; d++) {
              idx += mask_indices_accessor[d][i] * t.stride(d);
            }
            scalar_t val = t_ptr[idx];
            r_values_accessor[i] = val;
          }
    });
  }
}

SparseTensor sparse_mask_cpu(const Tensor& t, SparseTensorRef mask) {
  SparseTensor r = t.type().toSparse().tensor();
  sparse_mask_out_cpu(r, t, mask.tref);
  return r;
}

}} // namespace at::native
