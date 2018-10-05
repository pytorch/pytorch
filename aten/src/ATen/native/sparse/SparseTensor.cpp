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
  return detail::make_tensor<SparseTensorImpl>(
      type_id, scalarTypeToTypeMeta(dtype.scalarType()));
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

  const SparseType& dtype = values.type().toSparse();

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t sparseDims = indices.size(0);
  int64_t denseDims = values.dim() - 1;

  std::vector<int64_t> computed_sizes(sparseDims + denseDims);
  if (indices.numel() > 0) {
    // If the indices has elements in it, we infer the minimum sparse dimension sizes
    // as the max value of each dim in indices.
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
  } else {
    // If the indices doesn't have elements in it, there is not enough information
    // to know what the minimum sparse dimension sizes should be, and in this case
    // we set them to 0
    for (int64_t d = 0; d < sparseDims; d++) {
      computed_sizes[static_cast<size_t>(d)] = 0;
    }
  }
  for (int64_t d = 0; d < denseDims; d++) {
    computed_sizes[static_cast<size_t>(sparseDims + d)] = values.size(d+1);
  }
  return _new_with_dims_and_tensor_sparse(dtype, sparseDims, denseDims, computed_sizes, indices, values);
}

SparseTensor new_with_dims_and_size_sparse(const SparseType& dtype, int64_t sparseDims, int64_t denseDims, ArrayRef<int64_t> size) {
  SparseTensor self = new_sparse(dtype);
  AT_CHECK(size.size() != 0,
    "cannot construct sparse tensor with 0 dimensions and no values; you must specify at least 1 dimension if you want to create a sparse tensor with no elements, \
or you must provide a single-element `values` tensor (e.g. x = torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [])) if you want to create a scalar sparse tensor");
  _get_sparse_impl(self)->resize_and_clear_(sparseDims, denseDims, size);
  return self;
}

SparseTensor new_with_size_sparse(const SparseType& dtype, ArrayRef<int64_t> size) {
  return new_with_dims_and_size_sparse(dtype, size.size(), 0, size);
}

// NOTE: new_with_tensor_and_size_unsafe_sparse() differs from new_with_tensor_and_size_sparse()
// in that we don't check whether any indices are out of boundaries of `sizes`, thus avoiding a
// copy from CUDA to CPU. However, this function should ONLY be used where we know that the indices
// are guaranteed to be within bounds.
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

  int64_t sparseDims = indices.size(0);
  int64_t denseDims = values.dim() - 1;
  AT_CHECK(sizes.size() == sparseDims + denseDims, "number of dimensions must be sparseDims (", sparseDims, ") + denseDims (", denseDims, "), but got ", sizes.size());

  // Check to make sure all indices are within the boundaries of `sizes`
  if (indices.numel() > 0) {
    LongTensor min_indices = std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    LongTensor max_indices = std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    LongTensor cpu_min_indices, cpu_max_indices;
    if (indices.is_cuda()) {
      cpu_min_indices = at::CPU(kLong).copy(min_indices);
      cpu_max_indices = at::CPU(kLong).copy(max_indices);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
    for (int64_t d = 0; d < sparseDims; d++) {
      // NB: This used to sync ndim times to access each entry; now we copy
      // everything to CPU first and then access it.
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      AT_CHECK(min_index_in_dim >= 0,
               "found negative index ", min_index_in_dim, " for dim ", d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = sizes[static_cast<size_t>(d)];
      AT_CHECK(max_index_in_dim < dim_size,
               "sizes is inconsistent with indices: for dim ", d, ", size is ", dim_size, " but found index ", max_index_in_dim);
    }
  }
  return _new_with_dims_and_tensor_sparse(dtype, sparseDims, denseDims, sizes, indices, values);
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(const SparseTensor& self) {
  SparseTensor other = new_with_dims_and_size_sparse(self.type(), self._sparseDims(), self._denseDims(), self.sizes());
  _copy_into_sparse(other, _get_sparse_impl(self)->indices(), _get_sparse_impl(self)->values(), true);
  _get_sparse_impl(other)->set_coalesced(self.is_coalesced());
  return other;
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

SparseTensor& sparse_resize_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparseDims, int64_t denseDims) {
  _get_sparse_impl(self)->resize_(sparseDims, denseDims, size);
  return self;
}

SparseTensor& sparse_resize_and_clear_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparseDims, int64_t denseDims) {
  _get_sparse_impl(self)->resize_and_clear_(sparseDims, denseDims, size);
  return self;
}

namespace {
  bool _is_same_size_as_sparse(const SparseTensor& self, const SparseTensor& src) {
    return self._sparseDims() == src._sparseDims() && self._denseDims() == src._denseDims() && self.sizes().equals(src.sizes());
  }
}

SparseTensor& resize_as_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src._sparseDims(), src._denseDims());
  }
  return self;
}

// NB: Dropped the resizeNd variants

Tensor sparse_to_dense(const SparseTensor& self) {
  Tensor dst = at::zeros(self.sizes(), self.type().toDense());
  return dst.add_(self);
}

SparseTensor& copy_sparse_(SparseTensor& self, const SparseTensor& src, bool non_blocking) {
  if (isSameTensor(self, src)) return self;
  _get_sparse_impl(self)->resize_(src._sparseDims(), src._denseDims(), src.sizes());
  // NB: This seems to copy the underlying full indices/values buffer
  _copy_into_sparse(self, _get_sparse_impl(src)->indices(), _get_sparse_impl(src)->values(), non_blocking);
  _get_sparse_impl(self)->set_coalesced(src.is_coalesced());
  return self;
}

SparseTensor coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(self.defined());
  AT_ASSERT(!self.is_variable());
  AT_ASSERT(self.is_sparse());

  if (self.is_coalesced()) {
    return self;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (self._nnz() < 2) {
    SparseTensor dst = self.clone();
    _get_sparse_impl(dst)->set_coalesced(true);
    return dst;
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
  _get_sparse_impl(dst)->resize_(sparseDims, denseDims, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  LongTensor newIndices = at::empty(indices.sizes(), indices.options());
  Tensor newValues = at::empty(values.sizes(), values.options());
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
            if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
              THBlas_axpy<scalar_t>(blockSize, 1, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
            }
          } else {
            ++i;
            for (int64_t d = 0; d < sparseDims; d++) {
              newIndicesAccessor[d][i] = indicesAccessor[d][pos];
            }
            if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
              THBlas_copy<scalar_t>(blockSize, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
            }
          }
          prev = curr;
        }
    });

  _get_sparse_impl(dst)->set_coalesced(true);
  _get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

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
  Tensor r_values = at::empty(mask_values.sizes(), r._values().options());
  _alias_into_sparse(r, mask_indices.clone(), r_values);
  _get_sparse_impl(r)->set_coalesced(mask.is_coalesced());
  int64_t r_nnz = mask._nnz();
  _get_sparse_impl(r)->set_nnz_and_narrow(r_nnz);
  if (t.numel() == 0) {  // if t is an empty tensor, there is no need to mask its elements
    return r;
  }

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
  SparseTensor r = at::empty({0}, t.options().layout(kSparse));
  sparse_mask_out_cpu(r, t, mask.tref);
  return r;
}

}} // namespace at::native
