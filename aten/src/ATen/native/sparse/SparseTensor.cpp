// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

using namespace at::sparse;


/******************************************************************************
 * access methods
 ******************************************************************************/

int64_t sparse_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->sparse_dim();
}

int64_t dense_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->dense_dim();
}

bool is_coalesced_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->coalesced();
}

int64_t _nnz_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->nnz();
}

// Why are there so many methods to get indices and value?
// See Note [ Sparse: different methods to get indices and values ] in native_functions.yaml

Tensor _indices_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->indices();
}

Tensor _values_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->values();
}

Tensor &_coalesced_sparse_(SparseTensor& self, bool coalesced) {
  get_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

Tensor indices_sparse(const Tensor& self) {
  TORCH_CHECK(self.is_coalesced(),
           "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->indices().alias();
}

Tensor values_sparse(const Tensor& self) {
  TORCH_CHECK(self.is_coalesced(),
           "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->values().alias();
}

/******************************************************************************
 * creation methods
 * See NOTE [ Sparse: autograd and API ] for details
 ******************************************************************************/

/*** Helper methods ***/

SparseTensor new_sparse(const TensorOptions& options) {
  AT_ASSERT(!options.is_variable());  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERT(options.layout() == kSparse);
  TensorTypeId type_id;
  if (options.device().is_cuda()) {
    type_id = TensorTypeId::SparseCUDATensorId;
  } else {
    type_id = TensorTypeId::SparseCPUTensorId;
  }
  return detail::make_tensor<SparseTensorImpl>(
      TensorTypeSet(type_id), options.dtype());
}

/** Actual dispatched creation methods ***/

SparseTensor new_with_dims_sparse(int64_t sparse_dim, int64_t dense_dim, ArrayRef<int64_t> size, const TensorOptions& options) {
  SparseTensor self = new_sparse(options);
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

SparseTensor new_with_dims_and_tensor_sparse(
    int64_t sparse_dim,
    int64_t dense_dim,
    ArrayRef<int64_t> size,
    const LongTensor& indices,
    const Tensor& values,
    const TensorOptions& options) {
  SparseTensor self = new_sparse(options);
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  // NOTE: There is no guarantee that `indices` and `values` don't contain AutogradMeta. However,
  // we want to maintain the invariant that `indices_` and `values_` of a sparse tensor don't
  // contain AutogradMeta, and to achieve that we shallow-copy `indices` and `values` here.
  auto indices_shallow_copy = LongTensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
    /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy = Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
    /*allow_tensor_metadata_change=*/true));
  alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
Tensor empty_sparse(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  return new_with_dims_sparse(size.size(), 0, size, options);
}

/* Shape init */
Tensor sparse_coo_tensor(ArrayRef<int64_t> size, const TensorOptions& options) {
  return at::_sparse_coo_tensor_with_dims(size.size(), 0, size, options.layout(at::kSparse));
}

/* Pointer-copy init */

// helper
namespace {
  static inline Tensor expand_values_if_needed(const Tensor& values) {
    // expand
    if (values.dim() == 0) {
      // Mimic Numpy behavior here and treat it as a 1D tensor
      return values.expand({1});
    } else {
      return values;
    }
  }
}

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values_, const TensorOptions& options) {
  Tensor values = expand_values_if_needed(values_);

  // arg checking
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());
  // the following checks are redundant because they are also checked in SparseTensorImpl::set_indices_and_values_unsafe
  // but we need to ensure them in order to infer the shape.
  TORCH_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes())
  TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  std::vector<int64_t> computed_sizes(sparse_dim + dense_dim);
  if (indices.numel() > 0) {
    // If the indices has elements in it, we infer the minimum sparse dimension sizes
    // as the max value of each dim in indices.
    // NB: It used to keepdim. I think that was wrong.
    LongTensor min_indices = std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    LongTensor computed_indices_sizes = std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    computed_indices_sizes.add_(1); // len = max_index + 1
    LongTensor cpu_min_indices = min_indices.to(at::DeviceType::CPU);
    LongTensor cpu_computed_indices_sizes = computed_indices_sizes.to(at::DeviceType::CPU);
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_computed_indices_sizes_accessor = cpu_computed_indices_sizes.accessor<int64_t, 1>();
    for (int64_t d = 0; d < sparse_dim; d++) {
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(min_index_in_dim >= 0,
               "found negative index ", min_index_in_dim, " for dim ", d);
      computed_sizes[static_cast<size_t>(d)] = cpu_computed_indices_sizes_accessor[d];
    }
  } else {
    // If the indices doesn't have elements in it, there is not enough information
    // to know what the minimum sparse dimension sizes should be, and in this case
    // we set them to 0
    for (int64_t d = 0; d < sparse_dim; d++) {
      computed_sizes[static_cast<size_t>(d)] = 0;
    }
  }
  for (int64_t d = 0; d < dense_dim; d++) {
    computed_sizes[static_cast<size_t>(sparse_dim + d)] = values.size(d+1);
  }

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim, dense_dim, computed_sizes, indices, values, values.options().layout(kSparse));
}

// NB: Got rid of the sizes == NULL case
Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values_, ArrayRef<int64_t> size, const TensorOptions& options) {
  Tensor values = expand_values_if_needed(values_);

  // arg checking
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());
  // the following checks are redundant because they are also checked in SparseTensorImpl::set_indices_and_values_unsafe
  // but we need to ensure them in order to infer the shape.
  TORCH_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes())
  TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  TORCH_CHECK(size.size() == sparse_dim + dense_dim,
           "number of dimensions must be sparse_dim (", sparse_dim, ") + dense_dim (", dense_dim, "), but got ", size.size());

  // Check to make sure all indices are within the boundaries of `size`
  if (indices.numel() > 0) {
    LongTensor min_indices = std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    LongTensor max_indices = std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    LongTensor cpu_min_indices, cpu_max_indices;
    if (indices.is_cuda()) {
      cpu_min_indices = min_indices.to(at::DeviceType::CPU);
      cpu_max_indices = max_indices.to(at::DeviceType::CPU);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
    for (int64_t d = 0; d < sparse_dim; d++) {
      // NB: This used to sync ndim times to access each entry; now we copy
      // everything to CPU first and then access it.
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(min_index_in_dim >= 0,
               "found negative index ", min_index_in_dim, " for dim ", d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = size[static_cast<size_t>(d)];
      TORCH_CHECK(max_index_in_dim < dim_size,
               "size is inconsistent with indices: for dim ", d, ", size is ", dim_size, " but found index ", max_index_in_dim);
    }
  }

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim, dense_dim, size, indices, values, values.options().layout(kSparse));
}

// NOTE: _sparse_coo_tensor_unsafe() differs from sparse_coo_tensor()
// in that we don't check whether any indices are out of boundaries of `size`, thus avoiding a
// copy from CUDA to CPU. However, this function should ONLY be used where we know that the indices
// are guaranteed to be within bounds.
// NB: Got rid of the size == NULL case
Tensor _sparse_coo_tensor_unsafe(const Tensor& indices, const Tensor& values_, ArrayRef<int64_t> size, const TensorOptions& options) {
  Tensor values = expand_values_if_needed(values_);

  // arg checking
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());

  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim, dense_dim, size, indices, values, values.options().layout(kSparse));
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(const SparseTensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  SparseTensor other = new_with_dims_sparse(self.sparse_dim(), self.dense_dim(), self.sizes(), self.options());
  copy_into_sparse(other, self._indices(), self._values(), true);
  return other._coalesced_(self.is_coalesced());
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

SparseTensor& sparse_resize_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  return self;
}

SparseTensor& sparse_resize_and_clear_(SparseTensor& self, ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

namespace {
  bool _is_same_size_as_sparse(const SparseTensor& self, const SparseTensor& src) {
    return self.sparse_dim() == src.sparse_dim() && self.dense_dim() == src.dense_dim() && self.sizes().equals(src.sizes());
  }
}

// Invoked from native/Resize.cpp (no dynamic dispatch necessary)
SparseTensor& resize_as_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
  }
  return self;
}

SparseTensor dense_to_sparse(const Tensor& self){
  return dense_to_sparse(self, self.dim());
}

SparseTensor dense_to_sparse(const Tensor& self, int64_t sparse_dim){
  int64_t dims = self.dim();
  // TODO: it seems like sparse_dim == 0 could be supported even if self.dim() > 0,
  // but this would take some work and doesn't seem particularly useful.
  TORCH_CHECK(sparse_dim > 0 || self.dim() == 0, "sparse_dim must be >0 if dimensionality > 0");
  TORCH_CHECK(sparse_dim <= dims,
    "sparse_dim must be less than or equal to self.dim()");
  at::TensorOptions sparse_options = self.options().layout(kSparse);
  std::vector<int64_t> sizes = self.sizes().vec();

  Tensor nz = self.nonzero().transpose(0, 1);
  if (nz.size(1) == 0) {
    return new_with_dims_sparse(sparse_dim, dims - sparse_dim, sizes, sparse_options);
  }
  LongTensor indices;
  if (sparse_dim == dims) {
    indices = nz.clone();
  } else {
    Tensor i = nz.narrow(0, 0, sparse_dim);
    std::tie(indices, std::ignore, std::ignore) = unique_dim(i, 1);
    indices = indices.contiguous();  // many sparse CUDA kernels require contiguity, see issue #12633
  }

  Tensor values;
  if (self.dim() > 0) {
    std::vector<Tensor> ix = indices.chunk(indices.size(0), 0);
    values = self.index(ix).squeeze(0).clone(at::MemoryFormat::Preserve);
  } else {
    AT_ASSERT(nz.sizes().equals({0, 1}));
    // In this cases, indices is a clone of nz, which is a tensor of shape (0, 1).
    // Given sparse tensor invariants, values should be shape (1,)
    values = self.unsqueeze(0).clone(at::MemoryFormat::Preserve);
  }

  Tensor sparse = at::sparse_coo_tensor(indices, values, sizes, sparse_options);
  return sparse._coalesced_(true);
}

// NB: Dropped the resizeNd variants

Tensor sparse_to_dense(const SparseTensor& self) {
  if(self.scalar_type() == ScalarType::Half && self.options().device().is_cpu()) {
    AT_ERROR("to_dense() not supported for float16 on CPU");
  }
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

SparseTensor& copy_sparse_(SparseTensor& self, const SparseTensor& src, bool non_blocking) {
  if (is_same_tensor(self, src)) return self;
  get_sparse_impl(self)->resize_(src.sparse_dim(), src.dense_dim(), src.sizes());
  copy_into_sparse(self, src._indices(), src._values(), non_blocking);
  return self._coalesced_(src.is_coalesced());
}

SparseTensor coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(self.defined());
  AT_ASSERT(!self.is_variable());  // TODO: change this to check `.requires_grad()` and `GradMode::is_enabled()` when Variable and Tensor are merged
  AT_ASSERT(self.is_sparse());

  if (self.is_coalesced()) {
    return self;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (self._nnz() < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  LongTensor indices = self._indices();
  Tensor values = self._values().contiguous();
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  int64_t nnz = self._nnz();

  LongTensor indices_scalar = flatten_indices(indices, self.sizes());

  SparseTensor dst = new_sparse(self.options());
  get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  LongTensor newIndices = at::empty(indices.sizes(), indices.options());
  Tensor newValues = at::empty(values.sizes(), values.options());
  alias_into_sparse(dst, newIndices, newValues);

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
      values.scalar_type(), "coalesce", [&] {
        int64_t prev = -1;
        int64_t blockSize = values.stride(0);
        scalar_t* values_ptr = values.data_ptr<scalar_t>();
        scalar_t* newValues_ptr = newValues.data_ptr<scalar_t>();
        for (int64_t j = 0; j < nnz; j++) {
          int64_t pos = indicesPermutationAccessor[j];
          int64_t curr = indicesBufferAccessor[j];
          if (curr == prev) {
            if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
              THBlas_axpy<scalar_t>(blockSize, 1, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
            }
          } else {
            ++i;
            for (int64_t d = 0; d < sparse_dim; d++) {
              newIndicesAccessor[d][i] = indicesAccessor[d][pos];
            }
            if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
              THBlas_copy<scalar_t>(blockSize, values_ptr + pos * blockSize, 1, newValues_ptr + i * blockSize, 1);
            }
          }
          prev = curr;
        }
    });

  dst._coalesced_(true);
  get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

  return dst;
}


// --------------------------------------------------------------------
// sparse_mask(D, S) -> S
//
// Filter Tensor D by S.indices() and output a SparseTensor.
// D and S must share the same shape.
// --------------------------------------------------------------------

template <typename scalar_t>
void inline sparse_mask_out_cpu_kernel(
  Tensor& r_values,
  const Tensor& t,
  const int64_t r_nnz,
  const int64_t sparse_dim,
  const LongTensor& mask_indices
) {
  auto r_values_accessor = r_values.accessor<scalar_t, 1>();
  auto mask_indices_accessor = mask_indices.accessor<int64_t, 2>();
  scalar_t* t_ptr = t.data_ptr<scalar_t>();

  at::parallel_for(0, r_nnz, 1000, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      int64_t idx = 0;
      for (int64_t d = 0; d < sparse_dim; d++) {
        idx += mask_indices_accessor[d][i] * t.stride(d);
      }
      r_values_accessor[i] = t_ptr[idx];
    }
  });
}

SparseTensor& sparse_mask_out_cpu(SparseTensor& r, const Tensor& t, const SparseTensor& mask) {
  TORCH_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
  TORCH_CHECK(mask.sizes().equals(t.sizes()), "sparse_mask: operands have incompatible sizes; self has size ",
      t.sizes(), " but mask has size ", mask.sizes());
  AT_ASSERT(!t.is_cuda()); // we were supposed to have dispatched on this
  TORCH_CHECK(!r.is_cuda(), "sparse_mask: expected 'out' to be CPU, but got CUDA");
  TORCH_CHECK(!mask.is_cuda(), "sparse_mask: expected 'mask' to be CPU, but got CUDA");
  resize_as_sparse_(r, mask);
  if (mask._nnz() == 0) {
    return r.zero_();
  }
  int64_t dim = t.dim();
  int64_t sparse_dim = mask.sparse_dim();
  LongTensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = at::empty(mask_values.sizes(), r._values().options());
  alias_into_sparse(r, mask_indices.clone(), r_values);
  r._coalesced_(mask.is_coalesced());
  int64_t r_nnz = mask._nnz();
  get_sparse_impl(r)->set_nnz_and_narrow(r_nnz);

  if (t.numel() == 0) {  // if t is an empty tensor, there is no need to mask its elements
    return r;
  }

  if (dim > sparse_dim) {

    // Get a flattened sparse indices, similar to NOTE [ Flatten Sparse Indices ].
    // Keeping this implementation because it is faster than flatten_indices()
    LongTensor indices = at::zeros({mask._nnz()}, mask_indices.options());
    for (int64_t d = 0; d < mask.sparse_dim(); d++) {
      indices.mul_(mask.size(d));
      indices.add_(mask_indices.select(0, d));
    }

    std::vector<int64_t> view_size(1 + mask.dense_dim());
    view_size[0] = -1;
    for (int64_t d = 0; d < mask.dense_dim(); d++) {
      view_size[d + 1] = mask.size(mask.sparse_dim() + d);
    }

    Tensor t_view = t.view(view_size);
    // TODO: Re-audit this; it used to be an indexSelect directly into r_values
    at::index_select_out(r_values, t_view, 0, indices);
  } else {
    AT_DISPATCH_ALL_TYPES(r_values.scalar_type(), "sparse_mask", [&] {
      sparse_mask_out_cpu_kernel<scalar_t>(
        r_values,
        t,
        r_nnz,
        sparse_dim,
        mask_indices);
    });
  }
  return r;
}

SparseTensor sparse_mask_cpu(const Tensor& t, const SparseTensor& mask) {
  SparseTensor r = at::empty({0}, t.options().layout(kSparse));
  sparse_mask_out_cpu(r, t, mask);
  return r;
}

}} // namespace at::native
