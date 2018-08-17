// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseUtils.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

/******************************************************************************
 * access methods
 ******************************************************************************/

int64_t _sparseDims_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->sparseDims();
}

int64_t _denseDims_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->denseDims();
}

bool is_coalesced_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->coalesced();
}

int64_t _nnz_sparse(const SparseTensor& self) {
  return _get_sparse_impl(self)->nnz();
}

// TODO: This is wrong: if nnz == 0 but indices/values is not
// empty then we'll return all the values, even the ones that
// are "masked out" by nnz

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
  AT_ASSERT(!dtype.is_undefined());
  AT_ASSERT(!dtype.is_variable());
  AT_ASSERT(dtype.is_sparse());
  TensorTypeId type_id;
  if (dtype.is_cuda()) {
    type_id = SparseCUDATensorId();
  } else {
    type_id = SparseCPUTensorId();
  }
  return SparseTensor(new SparseTensorImpl(type_id, dtype.scalarType()), /* retain */ false);
}

/*** Helper methods ***/

/* Pointer-copy init */
SparseTensor new_with_tensor_sparse(const LongTensor& indices, const Tensor& values_) {
  Tensor values;
  if (values_.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    values = values_.expand({1});
  } else {
    values = values_;
  }

  // TODO: This is a temporary test until we support zero-size dims.
  // I'm NOT adding the "obvious" bypass code, because it wasn't supported
  // previously
  AT_CHECK(indices.numel() != 0, "cannot construct sparse tensor with empty indices; use the nullary constructor instead");

  const SparseType& dtype = values.type().toSparse();

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t sparseDims = indices.size(0);
  int64_t denseDims = values.dim() - 1;

  std::vector<int64_t> computed_sizes(sparseDims + denseDims);
  // NB: It used to keepdim. I think that was wrong.
  LongTensor computed_indices_sizes = std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
  computed_indices_sizes.add_(1); // len = max_index + 1
  LongTensor cpu_computed_indices_sizes;
  if (computed_indices_sizes.is_cuda()) {
    cpu_computed_indices_sizes = at::CPU(kLong).tensor(computed_indices_sizes.sizes());
    cpu_computed_indices_sizes.copy_(computed_indices_sizes);
  } else {
    cpu_computed_indices_sizes = computed_indices_sizes;
  }
  auto cpu_computed_indices_sizes_accessor = cpu_computed_indices_sizes.accessor<int64_t, 1>();
  for (int64_t d = 0; d < sparseDims; d++) {
    computed_sizes[static_cast<size_t>(d)] = cpu_computed_indices_sizes_accessor[d];
  }
  for (int64_t d = 0; d < denseDims; d++) {
    computed_sizes[static_cast<size_t>(sparseDims + d)] = values.size(d+1);
  }
  return _new_with_dims_and_tensor_sparse(dtype, sparseDims, denseDims, computed_sizes, indices, values);
}

SparseTensor new_with_size_sparse(const SparseType& dtype, ArrayRef<int64_t> size) {
  SparseTensor self = new_sparse(dtype);
  _raw_resize_sparse(self, size.size(), 0, size);
  return self;
}

// NB: Got rid of the sizes == NULL case
SparseTensor new_with_tensor_and_size_unsafe_sparse(const LongTensor& indices, const Tensor& values_, ArrayRef<int64_t> sizes) {
  Tensor values;
  if (values_.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    values = values_.expand({1});
  } else {
    values = values_;
  }

  const SparseType& dtype = values.type().toSparse();
  // NB: used to be a dim() == 0 test, but that's legacy TH semantics
  if (indices.numel() == 0 && values.numel() == 0) {
    return new_with_size_sparse(dtype, sizes);
  }

  int64_t sparseDims = indices.size(0);
  int64_t denseDims = values.dim() - 1;
  return _new_with_dims_and_tensor_sparse(dtype, sparseDims, denseDims, sizes, indices, values);
}

// NB: Got rid of the sizes == NULL case
SparseTensor new_with_tensor_and_size_sparse(const LongTensor& indices, const Tensor& values_, ArrayRef<int64_t> sizes) {
  Tensor values;
  if (values_.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    values = values_.expand({1});
  } else {
    values = values_;
  }

  const SparseType& dtype = values.type().toSparse();
  // NB: This used to be dims, but mumble TH handling zero-sized tensors
  // incorrectly
  if (indices.numel() == 0 && values.numel() == 0) {
    return new_with_size_sparse(dtype, sizes);
  }

  int64_t sparseDims = indices.size(0);
  int64_t denseDims = values.dim() - 1;
  AT_CHECK(sizes.size() == sparseDims + denseDims, "number of dimensions must be sparseDims (", sparseDims, ") + denseDims (", denseDims, "), but got ", sizes);

  LongTensor max_indices = std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
  LongTensor cpu_max_indices;
  if (max_indices.is_cuda()) {
    cpu_max_indices = at::CPU(kLong).copy(max_indices);
  } else {
    cpu_max_indices = max_indices;
  }
  auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
  for (int64_t d = 0; d < sparseDims; d++) {
    // NB: This used to sync ndim times to access each entry; now we copy
    // everything to CPU first and then access it.
    int64_t max_index_in_dim = cpu_max_indices_accessor[d];
    int64_t dim_size = sizes[static_cast<size_t>(d)];
    AT_CHECK(max_index_in_dim < dim_size,
             "sizes is inconsistent with indices: for dim ", d, ", size is ", dim_size, " but found index ", max_index_in_dim);
  }
  for (int64_t d = 0; d < denseDims; d++) {
    int64_t values_size = values.size(d+1);
    int64_t specified_size = sizes[static_cast<size_t>(sparseDims + d)];
    AT_CHECK(values_size <= specified_size,
             "values and sizes are inconsistent: sizes[", d + sparseDims, "] is ", specified_size,
             " but values.size(", d + 1, ") is ", values_size);
  }
  return _new_with_dims_and_tensor_sparse(dtype, sparseDims, denseDims, sizes, indices, values);
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(const SparseTensor& self) {
  SparseTensor other = new_sparse(self.type());
  _raw_resize_sparse(other, self._sparseDims(), self._denseDims(), self.sizes());
  // NB: This seems to preserve the size of the UN-narrowed indices and
  // values.  Veeery interesting.
  _copy_into_sparse(other, _get_sparse_impl(self)->indices(), _get_sparse_impl(self)->values());
  _get_sparse_impl(other)->set_coalesced(self.is_coalesced());
  _get_sparse_impl(other)->set_nnz(self._nnz());
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

SparseTensor& raw_resize_sparse_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparseDims, int64_t denseDims) {
  if (sparseDims == -1) {
    sparseDims = self._indices().size(0);
  }
  if (denseDims == -1) {
    denseDims = self._values().dim() - 1;
  }
  _raw_resize_sparse(self, sparseDims, denseDims, size);
  return self;
}

namespace {
  bool _is_same_size_as_sparse(const SparseTensor& self, const SparseTensor& src) {
    return self._sparseDims() == src._sparseDims() && self._denseDims() == src._denseDims() && self.sizes().equals(src.sizes());
  }
}

SparseTensor& resize_as_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    _raw_resize_sparse(self, src._sparseDims(), src._denseDims(), src.sizes());
  }
  return self;
}

// NB: Dropped the resizeNd variants

Tensor sparse_to_dense(const SparseTensor& self) {
  Tensor dst = at::zeros(self.sizes(), self.type().toDense());
  return dst.add_(self);
}

SparseTensor& copy_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (isSameTensor(self, src)) return self;
  _raw_resize_sparse(self, src._sparseDims(), src._denseDims(), src.sizes());
  // NB: This seems to copy the underlying full indices/values buffer
  _copy_into_sparse(self, _get_sparse_impl(src)->indices(), _get_sparse_impl(src)->values());
  _get_sparse_impl(self)->set_coalesced(src.is_coalesced());
  _get_sparse_impl(self)->set_nnz(src._nnz());
  return self;
}

SparseTensor coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(self.defined());
  AT_ASSERT(!self.is_variable());
  AT_ASSERT(self.is_sparse());

  if (self._nnz() < 2) {
    _get_sparse_impl(self)->set_coalesced(true);
  }
  if (self.is_coalesced()) {
    return self;
  }

  LongTensor indices = self._indices();
  Tensor values = self._values().contiguous();
  int64_t sparseDims = self._sparseDims();
  int64_t denseDims = self._denseDims();
  int64_t nnz = self._nnz();

  LongTensor indices_scalar = at::zeros({nnz}, kLong);

  int64_t factor = 1;
  for (int64_t d = sparseDims - 1; d >= 0; d--) {
    LongTensor indices_slice = indices.select(0, d);
    indices_scalar.add_(indices_slice, factor); // cadd is swapped args
    factor *= self.size(d);
  }

  SparseTensor dst = new_sparse(self.type());
  _raw_resize_sparse(dst, sparseDims, denseDims, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  LongTensor newIndices = indices.type().tensor(indices.sizes());
  Tensor newValues = values.type().tensor(values.sizes());
  _alias_into_sparse(dst, newIndices, newValues);

  LongTensor indicesBuffer;
  LongTensor indicesPermutation;
  std::tie(indicesBuffer, indicesPermutation) = indices_scalar.sort(0);
  // NB: The accessor accesses here rely on self._nnz() > 0 (tested earlier in this function)
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
            THBlas_axpy<scalar_t>(blockSize, 1, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
          } else {
            ++i;
            for (int64_t d = 0; d < sparseDims; d++) {
              newIndicesAccessor[d][i] = indicesAccessor[d][pos];
            }
            THBlas_copy<scalar_t>(blockSize, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
          }
          prev = curr;
        }
    });

  _get_sparse_impl(dst)->set_coalesced(true);
  _get_sparse_impl(dst)->set_nnz(i + 1);

  return dst;
}

SparseTensor& sparse_mask_out_cpu(SparseTensor& r, const Tensor& t, const SparseTensor& mask) {
  AT_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
  AT_CHECK(mask.sizes().equals(t.sizes()), "sparse_mask: operands have incompatible sizes; self has size ",
      t.sizes(), " but mask has size ", mask.sizes());
  AT_ASSERT(!t.is_cuda()); // we were supposed to have dispatched on this
  AT_CHECK(!r.is_cuda(), "sparse_mask: expected 'out' to be CPU, but got CUDA");
  AT_CHECK(!mask.is_cuda(), "sparse_mask: expected 'mask' to be CPU, but got CUDA");
  resize_as_sparse_(r, mask);
  if (mask._nnz() == 0) {
    r.zero_();
    return r;
  }
  int64_t dim = t.dim();
  int64_t sparseDims = mask._sparseDims();
  LongTensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = r._values().type().tensor(mask_values.sizes());
  _alias_into_sparse(r, mask_indices.clone(), r_values);
  _get_sparse_impl(r)->set_coalesced(mask.is_coalesced());
  int64_t r_nnz = mask._nnz();
  _get_sparse_impl(r)->set_nnz(r_nnz);
  // NB: Relies on mask._nnz() == 0 test above
  auto mask_indices_accessor = mask_indices.accessor<int64_t, 2>();

  if (dim > sparseDims) {
    // NB: This used to reuse buffers, but I deoptimized it
    for (int64_t i = 0; i < r_nnz; i++) {
      Tensor srcBuffer = t;
      for (int64_t d = 0; d < sparseDims; d++) {
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
            for (int64_t d = 0; d < sparseDims; d++) {
              idx += mask_indices_accessor[d][i] * t.stride(d);
            }
            scalar_t val = t_ptr[idx];
            r_values_accessor[i] = val;
          }
    });
  }
  return r;
}

SparseTensor sparse_mask_cpu(const Tensor& t, SparseTensorRef mask) {
  SparseTensor r = t.type().toSparse().tensor();
  sparse_mask_out_cpu(r, t, mask.tref);
  return r;
}

}} // namespace at::native
