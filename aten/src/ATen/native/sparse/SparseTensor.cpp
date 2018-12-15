// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
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
  AT_CHECK(self.is_coalesced(),
           "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->indices().alias();
}

Tensor values_sparse(const Tensor& self) {
  AT_CHECK(self.is_coalesced(),
           "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->values().alias();
}

/******************************************************************************
 * creation methods
 * See NOTE [ Sparse: autograd and API ] for details
 ******************************************************************************/

/*** Helper methods ***/

SparseTensor new_sparse(const TensorOptions& options) {
  AT_ASSERT(!options.is_variable());
  AT_ASSERT(options.layout() == kSparse);
  TensorTypeId type_id;
  if (options.device().is_cuda()) {
    type_id = SparseCUDATensorId();
  } else {
    type_id = SparseCPUTensorId();
  }
  return detail::make_tensor<SparseTensorImpl>(
      type_id, options.dtype());
}

/** Actual dispatched creation methods ***/

SparseTensor new_with_dims_sparse(int64_t sparse_dim, int64_t dense_dim, ArrayRef<int64_t> size, const TensorOptions& options) {
  SparseTensor self = new_sparse(options);
  AT_CHECK(size.size() != 0,
    "cannot construct sparse tensor with 0 dimensions and no values; you must specify at least 1 dimension if you want to create a sparse tensor with no elements, \
or you must provide a single-element `values` tensor (e.g. x = torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [])) if you want to create a scalar sparse tensor");
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

// Does NOT make copies of indices and values
SparseTensor new_with_dims_and_tensor_sparse(
    int64_t sparse_dim,
    int64_t dense_dim,
    ArrayRef<int64_t> size,
    const LongTensor& indices,
    const Tensor& values,
    const TensorOptions& options) {
  SparseTensor self = new_sparse(options);
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  alias_into_sparse(self, indices, values);
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
Tensor empty_sparse(IntList size, const TensorOptions& options) {
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
  AT_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());
  // the following checks are redundant because they are also checked in SparseTensorImpl::set_indices_and_values_unsafe
  // but we need to ensure them in order to infer the shape.
  AT_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes())
  AT_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());

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
      AT_CHECK(min_index_in_dim >= 0,
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
  AT_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());
  // the following checks are redundant because they are also checked in SparseTensorImpl::set_indices_and_values_unsafe
  // but we need to ensure them in order to infer the shape.
  AT_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes())
  AT_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  AT_CHECK(size.size() == sparse_dim + dense_dim,
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
      AT_CHECK(min_index_in_dim >= 0,
               "found negative index ", min_index_in_dim, " for dim ", d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = size[static_cast<size_t>(d)];
      AT_CHECK(max_index_in_dim < dim_size,
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
  AT_CHECK(!options.has_layout() || options.layout() == kSparse, "expected sparse layout, but got layout ", options.layout());

  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim, dense_dim, size, indices, values, values.options().layout(kSparse));
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(const SparseTensor& self) {
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

SparseTensor& resize_as_sparse_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
  }
  return self;
}

// --------------------------------------------------------------------
// conversion between sparse and dense
// --------------------------------------------------------------------
SparseTensor dense_to_sparse(const Tensor& self){
  return dense_to_sparse(self, self.dim());
}

SparseTensor dense_to_sparse(const Tensor& self, int64_t sparse_dim){
  int64_t dims = self.dim();
  AT_CHECK(sparse_dim > 0, "sparse_dim must be >0");
  AT_CHECK(sparse_dim <= dims,
    "sparse_dim must be less than or equal to self.dim()");
  at::TensorOptions sparse_options = self.options().layout(kSparse);
  std::vector<int64_t> sizes = self.sizes().vec();

  Tensor nz = self.nonzero().transpose(0, 1);
  if (nz.numel() == 0) {
    return new_with_dims_sparse(sparse_dim, dims - sparse_dim, sizes, sparse_options);
  }
  LongTensor indices;
  if (sparse_dim == dims) {
    indices = nz.clone();
  } else {
    Tensor i = nz.narrow(0, 0, sparse_dim);
    std::tie(indices, std::ignore) = _unique_dim(i, 1);
    indices = indices.contiguous();  // many sparse CUDA kernels require contiguity, see issue #12633
  }

  std::vector<Tensor> ix = indices.chunk(indices.size(0), 0);
  Tensor values = self.index(ix).squeeze(0).clone();

  Tensor sparse = at::sparse_coo_tensor(indices, values, sizes, sparse_options);
  return sparse._coalesced_(true);
}

// NB: Dropped the resizeNd variants

Tensor sparse_to_dense(const SparseTensor& self) {
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

// --------------------------------------------------------------------
// sparse copy from sparse
// --------------------------------------------------------------------
SparseTensor& copy_sparse_(SparseTensor& self, const SparseTensor& src, bool non_blocking) {
  if (is_same_tensor(self, src)) return self;
  get_sparse_impl(self)->resize_(src.sparse_dim(), src.dense_dim(), src.sizes());
  copy_into_sparse(self, src._indices(), src._values(), non_blocking);
  return self._coalesced_(src.is_coalesced());
}

// -------------------------------------------------------------------------------------------
// NOTE [Sparse Coalesce]
//
// Coalescing a SparseTensor typically will combine duplicates (indices and values),
// and output a coalesced SparseTensor, where:
// 1. indices are unique
// 2. indices are sorted
//
// This is also called `coalesce_sum`, where the reduction is doing a sum operation.
// Similarly, there are `coalesce_max` and `coalesce_min` where the reduction operations
// are max and min.
//
// Depends on the input, coalesce will do the following:
// 1. if input SparseTensor is already coalesced, return input
// 2. if input has nnz <= 1, then it must be coalesced, return a clone of it with
//    `is_coalesced = true`. It is because `coalesce` is not an in-place operation when
//    `is_coalesced` is false, we should keep the original tensor intact and do coalesce
//     on a copy of the tensor
// 3. if input has nnz > 1, do the actual coalesce reduction (sum, max, min)
//    and return a coalesced tensor
// -------------------------------------------------------------------------------------------

// --------------------------------------------------------------------
// coalesce sum
// --------------------------------------------------------------------
template <typename scalar_t>
void coalesce_sum_op_cpu(int64_t n, scalar_t* a, scalar_t* b) {
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    b[i] += a[i];
  }
}

template <typename scalar_t>
void coalesce_copy_op_cpu(int64_t n, scalar_t* a, scalar_t* b) {
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

template <typename scalar_t, typename func_t>
void coalesce_sum_kernel_cpu(
  Tensor& values,
  Tensor& indices,
  Tensor& indices_permutation,
  Tensor& indices_buffer,
  Tensor& new_values,
  Tensor& new_indices,
  Tensor& new_nnz,
  int64_t sparse_dim,
  int64_t nnz,
  const func_t& reduce_op
) {
  int64_t prev = -1;
  int64_t block_size = values.stride(0);
  int64_t* new_nnz_ptr = new_nnz.data<int64_t>();
  new_nnz_ptr[0] = -1;
  scalar_t* values_ptr = values.data<scalar_t>();
  scalar_t* new_values_ptr = new_values.data<scalar_t>();
  auto new_indices_accessor = new_indices.accessor<int64_t, 2>();
  auto indicesAccessor = indices.accessor<int64_t, 2>();
  auto indices_permutation_accessor = indices_permutation.accessor<int64_t, 1>();
  auto indices_buffer_accessor = indices_buffer.accessor<int64_t, 1>();

  for (int64_t j = 0; j < nnz; j++) {
    int64_t pos = indices_permutation_accessor[j];
    int64_t curr = indices_buffer_accessor[j];
    if (curr == prev) {
      if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
        reduce_op(block_size, values_ptr + pos * block_size, new_values_ptr + new_nnz_ptr[0] * block_size);
      }
    } else {
      new_nnz_ptr[0] += 1;
      for (int64_t d = 0; d < sparse_dim; d++) {
        new_indices_accessor[d][new_nnz_ptr[0]] = indicesAccessor[d][pos];
      }
      if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
        coalesce_copy_op_cpu<scalar_t>(block_size, values_ptr + pos * block_size, new_values_ptr + new_nnz_ptr[0] * block_size);
      }
    }
    prev = curr;
  }
  new_nnz_ptr[0] += 1;
}

SparseTensor coalesce_sum_cpu(const SparseTensor& self) {
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
  LongTensor new_indices = at::empty_like(indices);
  Tensor new_values = at::empty_like(values);
  alias_into_sparse(dst, new_indices, new_values);

  LongTensor indices_buffer;
  LongTensor indices_permutation;
  std::tie(indices_buffer, indices_permutation) = indices_scalar.sort(0);

  auto new_nnz = at::empty({1}, indices.options());

  AT_DISPATCH_ALL_TYPES(values.type(), "coalesce_sum_cpu", [&] {
    coalesce_sum_kernel_cpu<scalar_t>(
      values,
      indices,
      indices_permutation,
      indices_buffer,
      new_values,
      new_indices,
      new_nnz,
      sparse_dim,
      nnz,
      coalesce_sum_op_cpu<scalar_t>
    );
  });

  dst._coalesced_(true);
  get_sparse_impl(dst)->set_nnz_and_narrow(new_nnz.data<int64_t>()[0]);

  return dst;
}

SparseTensor coalesce_sparse_cpu(const SparseTensor& self) {
  return coalesce_sum_cpu(self);
}

// --------------------------------------------------------------------
// coalesce max / min
// --------------------------------------------------------------------
template <typename scalar_t>
void coalesce_max_op_cpu(int64_t n, scalar_t* a, scalar_t* b,
  TensorAccessor<int64_t, 2>& reduction_indices_accessor,
  int64_t new_nnz_i, int64_t nnz_i
) {
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    if (a[i] > b[i]) {
      b[i] = a[i];
      reduction_indices_accessor[new_nnz_i][i] = nnz_i;
    }
  }
}

template <typename scalar_t>
void coalesce_min_op_cpu(int64_t n, scalar_t* a, scalar_t* b,
  TensorAccessor<int64_t, 2>& reduction_indices_accessor,
  int64_t new_nnz_i, int64_t nnz_i
) {
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    if (a[i] < b[i]) {
      b[i] = a[i];
      reduction_indices_accessor[new_nnz_i][i] = nnz_i;
    }
  }
}

template <typename scalar_t>
void coalesce_maxmin_copy_op_cpu(int64_t n, scalar_t* a, scalar_t* b,
  TensorAccessor<int64_t, 2>& reduction_indices_accessor,
  int64_t new_nnz_i, int64_t nnz_i
) {
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    b[i] = a[i];
    reduction_indices_accessor[new_nnz_i][i] = nnz_i;
  }
}

template <typename scalar_t, typename func_t>
void coalesce_maxmin_kernel_cpu(
  Tensor& values,
  Tensor& indices,
  Tensor& indices_permutation,
  Tensor& indices_buffer,
  Tensor& new_values,
  Tensor& new_indices,
  Tensor& reduction_indices,
  Tensor& new_nnz,
  int64_t sparse_dim,
  int64_t nnz,
  const func_t& reduce_op
) {
  int64_t prev = -1;
  int64_t block_size = values.stride(0);
  int64_t* new_nnz_ptr = new_nnz.data<int64_t>();
  new_nnz_ptr[0] = -1;
  scalar_t* values_ptr = values.data<scalar_t>();
  scalar_t* new_values_ptr = new_values.data<scalar_t>();
  auto new_indices_accessor = new_indices.accessor<int64_t, 2>();
  auto reduction_indices_accessor = reduction_indices.accessor<int64_t, 2>();
  auto indicesAccessor = indices.accessor<int64_t, 2>();
  auto indices_permutation_accessor = indices_permutation.accessor<int64_t, 1>();
  auto indices_buffer_accessor = indices_buffer.accessor<int64_t, 1>();

  for (int64_t j = 0; j < nnz; j++) {
    int64_t pos = indices_permutation_accessor[j];
    int64_t curr = indices_buffer_accessor[j];
    if (curr == prev) {
      if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
        reduce_op(
          block_size,
          values_ptr + pos * block_size,
          new_values_ptr + new_nnz_ptr[0] * block_size,
          reduction_indices_accessor,
          new_nnz_ptr[0], pos
        );
      }
    } else {
      new_nnz_ptr[0] += 1;
      for (int64_t d = 0; d < sparse_dim; d++) {
        new_indices_accessor[d][new_nnz_ptr[0]] = indicesAccessor[d][pos];
      }
      for (int64_t d = 0; d < block_size; d++) {
        reduction_indices_accessor[new_nnz_ptr[0]][d] = j;
      }
      if (values.numel() > 0) {  // if values is an empty tensor, there are no elements to copy
        // copy & init both of values and indices
        coalesce_maxmin_copy_op_cpu<scalar_t>(
          block_size,
          values_ptr + pos * block_size,
          new_values_ptr + new_nnz_ptr[0] * block_size,
          reduction_indices_accessor,
          new_nnz_ptr[0], pos
        );
      }
    }
    prev = curr;
  }
  new_nnz_ptr[0] += 1;
}

std::tuple<SparseTensor, Tensor> coalesce_maxmin_common_cpu(const SparseTensor& self, CoalesceReductionType reduction_type) {
  AT_ASSERT(self.defined());
  AT_ASSERT(!self.is_variable());
  AT_ASSERT(self.is_sparse());

  int64_t nnz = self._nnz();
  LongTensor indices = self._indices();
  Tensor values = self._values().contiguous();

  // NOTE [Reduction Indices at Coalesce]
  //
  // A 2D dense tensor reduction_indices is created for backward of sparse reductions (max / min),
  // during which grad values will be copied to grad-of-input based on reduction_indices.
  // Specifically, reduction_indices is a 2D tensor with the same shape as grad.values() -
  // {reduced_nnz, input.values().stride(0)}, therefore each value at reduction_indices indicates
  // a location where a grad value should be copied into.
  //
  // 1. when input is already coalesced, no reduction is needed, grad and input tensors will share
  //    the same shape. During backward values of grad will be copied to grad-of-input at
  //    exact the same locations. Therefore reduction_indices is the following:
  //
  //   [
  //      [0, 0, 0, 0, ..., 0],
  //      [1, 1, 1, 1, ..., 1]
  //      ...
  //      [nnz-1, nnz-1, ...,]
  //   ]
  //
  // 2. when input is uncoalesced, actual reduction will be performed, each row of reduction_indices
  //    store locations of where the reduced values are coming from.
  //
  // Note that the 2nd dim of reduction_indices equals to input.values().stride(0), which is numel of
  // of the dense tensor at the hybrid SparseTensor input. The size of 2nd dim equals to 1 when dense dim = 0.
  LongTensor reduction_indices;

  if (self.is_coalesced()) {
    reduction_indices = at::arange(0, nnz, indices.options()).reshape({nnz, 1}).repeat({1, values.stride(0)});
    return std::tuple<SparseTensor, Tensor>(self, reduction_indices);
  }

  if (nnz < 2) {
    // see NOTE [Reduction Indices at Coalesce]
    reduction_indices = at::arange(0, nnz, indices.options()).reshape({nnz, 1}).repeat({1, values.stride(0)});
    // see NOTE [Coalesce SparseTensor]
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return std::tuple<SparseTensor, Tensor>(dst, reduction_indices);
  }

  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();

  LongTensor indices_scalar = flatten_indices(indices, self.sizes());

  SparseTensor dst = new_sparse(self.options());
  get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
  LongTensor new_indices = at::empty_like(indices);
  Tensor new_values = at::empty_like(values);
  alias_into_sparse(dst, new_indices, new_values);

  LongTensor indices_buffer;
  LongTensor indices_permutation;
  std::tie(indices_buffer, indices_permutation) = indices_scalar.sort(0);

  auto new_nnz = at::empty({1}, indices.options());
  reduction_indices = at::empty({nnz, values.stride(0)}, indices.options());

  AT_DISPATCH_ALL_TYPES(values.type(), "coalesce_maxmin_common_cpu", [&] {

    if (reduction_type == CoalesceReductionType::MAX) {
      coalesce_maxmin_kernel_cpu<scalar_t>(
        values,
        indices,
        indices_permutation,
        indices_buffer,
        new_values,
        new_indices,
        reduction_indices,
        new_nnz,
        sparse_dim,
        nnz,
        coalesce_max_op_cpu<scalar_t>
      );
    }
    else if (reduction_type == CoalesceReductionType::MIN) {
      coalesce_maxmin_kernel_cpu<scalar_t>(
        values,
        indices,
        indices_permutation,
        indices_buffer,
        new_values,
        new_indices,
        reduction_indices,
        new_nnz,
        sparse_dim,
        nnz,
        coalesce_min_op_cpu<scalar_t>
      );
    }
    else {
      AT_ERROR("expected CoalesceReductionType MAX and MIN, but other type is found.");
    }
  });

  dst._coalesced_(true);
  get_sparse_impl(dst)->set_nnz_and_narrow(new_nnz.data<int64_t>()[0]);
  reduction_indices = reduction_indices.narrow_copy(0, 0, new_nnz.data<int64_t>()[0]);
  return std::tuple<SparseTensor, Tensor>(dst, reduction_indices);
}

std::tuple<SparseTensor, Tensor> coalesce_max_cpu(const SparseTensor& self) {
  return coalesce_maxmin_common_cpu(self, CoalesceReductionType::MAX);
}

std::tuple<SparseTensor, Tensor> coalesce_min_cpu(const SparseTensor& self) {
  return coalesce_maxmin_common_cpu(self, CoalesceReductionType::MIN);
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
  scalar_t* t_ptr = t.data<scalar_t>();
  int64_t i;

  #pragma omp parallel for private(i) if (r_nnz > 1000)
  for (i = 0; i < r_nnz; i++) {
    int64_t idx = 0;
    for (int64_t d = 0; d < sparse_dim; d++) {
      idx += mask_indices_accessor[d][i] * t.stride(d);
    }
    r_values_accessor[i] = t_ptr[idx];
  }
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
    AT_DISPATCH_ALL_TYPES(r_values.type(), "sparse_mask", [&] {
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

SparseTensor sparse_mask_cpu(const Tensor& t, SparseTensorRef mask) {
  SparseTensor r = at::empty({0}, t.options().layout(kSparse));
  sparse_mask_out_cpu(r, t, mask.tref);
  return r;
}

}} // namespace at::native
