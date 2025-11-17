// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/NamedTensorUtils.h>

#include <ATen/native/Copy.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce.h>
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_dimI_native.h>
#include <ATen/ops/_dimV_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_pin_memory_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/coalesce_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/_sparse_mask_projection_native.h>
#include <ATen/ops/sparse_resize_and_clear_native.h>
#include <ATen/ops/sparse_resize_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/unique_dim.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#endif

namespace at::native {

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

bool is_coalesced_default(const Tensor& self) {
  TORCH_CHECK(false, "is_coalesced expected sparse coordinate tensor layout but got ", self.layout());
  return false;
}

int64_t _nnz_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->nnz();
}

// Why are there so many methods to get indices and value?
// See Note [ Sparse: different methods to get indices and values ] in
// native_functions.yaml

Tensor _indices_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->indices();
}

Tensor _values_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->values();
}

Tensor& _coalesced_sparse_(SparseTensor& self, bool coalesced) {
  get_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

Tensor indices_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->indices().alias();
}

Tensor indices_default(const Tensor& self) {
  TORCH_CHECK(false, "indices expected sparse coordinate tensor layout but got ", self.layout());
}

Tensor values_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->values().alias();
}

Tensor values_default(const Tensor& self) {
  TORCH_CHECK(false, "values expected sparse tensor layout but got ", self.layout());
}

/******************************************************************************
 * creation methods
 * See NOTE [ Sparse: autograd and API ] for details
 ******************************************************************************/

/*** Helper methods ***/

static SparseTensor new_sparse(
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  AT_ASSERT(layout.has_value() && *layout == kSparse);
  DispatchKey dispatch_key;
  switch (device_or_default(device).type()) {
#define DO_CASE(device, _) \
    case DeviceType::device: \
      dispatch_key = DispatchKey::Sparse##device; \
      break;
    C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
    default:
      TORCH_CHECK(false, "device type not supported for sparse ", device_or_default(device))
  }
  return detail::make_tensor<SparseTensorImpl>(
      DispatchKeySet(dispatch_key),
      scalarTypeToTypeMeta(dtype_or_default(dtype)));
}

/** Actual dispatched creation methods ***/

SparseTensor new_with_dims_sparse(
    int64_t sparse_dim,
    int64_t dense_dim,
    ArrayRef<int64_t> size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

SparseTensor new_with_dims_and_tensor_sparse_symint(
    int64_t sparse_dim,
    int64_t dense_dim,
    c10::SymIntArrayRef size,
    const Tensor& indices,
    const Tensor& values,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  auto impl = get_sparse_impl(self);
  impl->resize_(sparse_dim, dense_dim, size);
  // NOTE: There is no guarantee that `indices` and `values` don't contain
  // AutogradMeta. However, we want to maintain the invariant that `indices_`
  // and `values_` of a sparse tensor don't contain AutogradMeta, and to achieve
  // that we shallow-copy `indices` and `values` here.
  auto indices_shallow_copy =
      Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy =
      Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  if (pin_memory.value_or(false)) {
    alias_into_sparse(self, indices_shallow_copy.pin_memory(), values_shallow_copy.pin_memory());
  } else {
    alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
  }
  // alias_into_sparse overrides coalesced flag, so resetting the flag to
  // the desired state here:
  if (is_coalesced.has_value()) {
    impl->set_coalesced(*is_coalesced);
  }
  // TODO: alias_into_sparse sets the coalesce flag to
  // `self._values().shape[0] < 2`. There exist methods (e.g. permute
  // on COO tensors when `dims[0] != 0` holds) that force coalesced
  // flag to false even when nnz is less that 2. Here we cannot
  // determine if this is the intention of such methods but it is
  // likely that these methods are overly restrictive when estimating
  // is_coalesced state. The condition `!is_coalesced && self._nnz() <
  // 2` provides a way to detect and optimize such methods with
  // respect to estimating the is_coalesced state.
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
Tensor empty_sparse_symint(
    SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  // TODO: Don't specialize
  return empty_sparse(C10_AS_INTARRAYREF_SLOW_ALLOC(size), dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor empty_sparse(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !pin_memory.has_value() || !*pin_memory,
      "Only dense CPU tensors can be pinned");
  return new_with_dims_sparse(
      size.size(), 0, size, dtype, layout, device, pin_memory);
}

/* Shape init */
Tensor sparse_coo_tensor(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

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
} // namespace

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values_,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  Tensor values = expand_values_if_needed(values_);

  // arg checking
  TORCH_CHECK(
      !options.has_layout() || options.layout() == kSparse,
      "expected sparse layout, but got layout ",
      options.layout());
  // the following checks are redundant because they are also checked in
  // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
  // in order to infer the shape.
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  std::vector<int64_t> computed_sizes(sparse_dim + dense_dim);
  if (indices.numel() > 0) {
    // If the indices has elements in it, we infer the minimum sparse dimension
    // sizes as the max value of each dim in indices. NB: It used to keepdim. I
    // think that was wrong.
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor computed_indices_sizes =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    computed_indices_sizes.add_(1); // len = max_index + 1
    Tensor cpu_min_indices = min_indices.to(at::DeviceType::CPU);
    Tensor cpu_computed_indices_sizes =
        computed_indices_sizes.to(at::DeviceType::CPU);
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_computed_indices_sizes_accessor =
        cpu_computed_indices_sizes.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      computed_sizes[static_cast<size_t>(d)] =
          cpu_computed_indices_sizes_accessor[d];
    }
  } else {
    // If the indices doesn't have elements in it, there is not enough
    // information to know what the minimum sparse dimension sizes should be,
    // and in this case we set them to 0
    for (const auto d : c10::irange(sparse_dim)) {
      computed_sizes[static_cast<size_t>(d)] = 0;
    }
  }
  for (const auto d : c10::irange(dense_dim)) {
    computed_sizes[static_cast<size_t>(sparse_dim + d)] = values.size(d + 1);
  }

  return at::native::_sparse_coo_tensor_unsafe(
      indices,
      values,
      computed_sizes,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      is_coalesced);
}

void _validate_sparse_coo_tensor_args(
    const Tensor& indices,
    const Tensor& values_,
    ArrayRef<int64_t> size,
    std::optional<bool> is_coalesced_,
    std::optional<bool> check_pinning_) {
  Tensor values = expand_values_if_needed(values_);
  bool is_coalesced = is_coalesced_.value_or(false);
  const bool check_pinning = check_pinning_.value_or(true);

  // the following checks are redundant because they are also checked in
  // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
  // in order to infer the shape.
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  TORCH_CHECK(
    sparse_dim + dense_dim == static_cast<int64_t>(size.size()),
    "'len(size) == sparse_dim + dense_dim' is not satisfied: len(size) = ",
    size.size(),
    ", sparse_dim = ",
    sparse_dim,
    ", dense_dim = ",
    dense_dim);

  if (check_pinning) {
    TORCH_CHECK(
        indices.is_pinned() == values.is_pinned(),
        "memory pinning of indices (=",
        indices.is_pinned(),
        ") must match memory pinning of values (=",
        values.is_pinned(),
        ")");
  }

  // Check to make sure all indices are within the boundaries of `size`
  if (indices.numel() > 0) {
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor max_indices =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    Tensor cpu_min_indices, cpu_max_indices;
    if (!indices.is_cpu()) {
      cpu_min_indices = min_indices.to(at::DeviceType::CPU);
      cpu_max_indices = max_indices.to(at::DeviceType::CPU);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      // NB: This used to sync ndim times to access each entry; now we copy
      // everything to CPU first and then access it.
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = size[static_cast<size_t>(d)];
      TORCH_CHECK(
          max_index_in_dim < dim_size,
          "size is inconsistent with indices: for dim ",
          d,
          ", size is ",
          dim_size,
          " but found index ",
          max_index_in_dim);
    }
    if (is_coalesced && values.size(0) > 1) {
      Tensor indices_scalar = flatten_indices(indices, size);
      Tensor diff = indices_scalar.diff();
      TORCH_CHECK(diff.min().item().toLong() > 0, "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor");
    }
  }
}

// NB: Got rid of the sizes == NULL case
Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values, IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // arg checking
  TORCH_CHECK(
      !options.has_layout() || options.layout() == kSparse,
      "expected sparse layout, but got layout ",
      options.layout());
  return at::native::_sparse_coo_tensor_unsafe(
      indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      is_coalesced);
}

Tensor _sparse_coo_tensor_unsafe(const Tensor& indices, const Tensor& values_, at::IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  if (at::globalContext().checkSparseTensorInvariants()) {
    at::native::_validate_sparse_coo_tensor_args(indices, values_, size, is_coalesced);
  }
  return at::native::_sparse_coo_tensor_unsafe_symint(indices, values_, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory, is_coalesced);
}

// NOTE: _sparse_coo_tensor_unsafe() differs from sparse_coo_tensor()
// in that we don't check whether any indices are out of boundaries of `size`, thus avoiding a
// copy from CUDA to CPU. However, this function should ONLY be used where we know that the indices
// are guaranteed to be within bounds or if the caller is going to call
// _validate_sparse_coo_tensor_args before using the tensor.
// NB: Got rid of the size == NULL case
Tensor _sparse_coo_tensor_unsafe_symint(const Tensor& indices, const Tensor& values_, c10::SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // See [Note: hacky wrapper removal for TensorOptions]

  Tensor values = expand_values_if_needed(values_);

  // This guard is intentional: we don't support dynamic shapes along the
  // indices dimension because that implies variable dimensionality
  auto sparse_dim = indices.sym_size(0).guard_int(__FILE__, __LINE__);
  auto dense_dim = values.dim() - 1;
  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      values.options().layout(kSparse).pinned_memory(pin_memory),
      is_coalesced);
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(
    const SparseTensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  SparseTensor other = new_with_dims_sparse(
      self.sparse_dim(),
      self.dense_dim(),
      self.sizes(),
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  copy_into_sparse(other, self._indices(), self._values(), true);
  return other._coalesced_(self.is_coalesced());
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

const SparseTensor& sparse_resize_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  return self;
}

const SparseTensor& sparse_resize_and_clear_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

namespace {
bool _is_same_size_as_sparse(
    const SparseTensor& self,
    const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() &&
      self.dense_dim() == src.dense_dim() && self.sizes().equals(src.sizes());
}
} // namespace

// Invoked from native/Resize.cpp (no dynamic dispatch necessary)
const SparseTensor& resize_as_sparse_(const SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
  }
  return self;
}

// NB: Dropped the resizeNd variants

SparseTensor& copy_sparse_wrapper_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  // TODO: Once copy_ is fully migrated to use dispatcher, handle named
  // inference using dispatcher instead of doing it everywhere
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    if (!self.is_sparse() || !src.is_sparse()) {
      TORCH_CHECK(false,
          "copy_() between dense and sparse Tensors is not implemented! Found self type = ",
          self.toString(),
          " and src type = ",
          src.toString());
    }
    at::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

SparseTensor& copy_sparse_(
    SparseTensor& self,
    const SparseTensor& src,
    bool non_blocking) {
  if (is_same_tensor(self, src))
    return self;
  get_sparse_impl(self)->resize_(
      src.sparse_dim(), src.dense_dim(), src.sizes());
  copy_into_sparse(self, src._indices(), src._values(), non_blocking);
  return self._coalesced_(src.is_coalesced());
}

SparseTensor coalesce(const SparseTensor& self) {
  TORCH_CHECK(self.layout() == kSparse, "coalesce expected sparse coordinate tensor layout but got ", self.layout());
  // See NOTE: [ coalesce autograd ]
  if (self.is_coalesced()) {
    return self;
  }
  return at::_coalesce(self);
}

SparseTensor _coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  AT_ASSERT(self.is_sparse());
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());

  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (self._nnz() < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor indices = self._indices();
  Tensor values = self._values().contiguous();
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  int64_t nnz = self._nnz();

  Tensor indices_scalar = flatten_indices(indices, self.sizes());

  SparseTensor dst = new_sparse(
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  Tensor newIndices = at::empty(indices.sizes(), indices.options());
  Tensor newValues = at::empty(values.sizes(), values.options());
  alias_into_sparse(dst, newIndices, newValues);

  auto [indicesBuffer, indicesPermutation] = indices_scalar.sort(0);
  // NB: The accessor accesses here rely on self._nnz() > 0 (tested earlier in
  // this function)
  auto newIndicesAccessor = newIndices.accessor<int64_t, 2>();
  auto indicesAccessor = indices.accessor<int64_t, 2>();
  auto indicesPermutationAccessor = indicesPermutation.accessor<int64_t, 1>();
  auto indicesBufferAccessor = indicesBuffer.accessor<int64_t, 1>();

  int64_t i = -1;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf, at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Bool,
      values.scalar_type(), "coalesce", [&] {
    int64_t prev = -1;
    int64_t blockSize = values.stride(0);
    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    scalar_t* newValues_ptr = newValues.data_ptr<scalar_t>();
    for (const auto j : c10::irange(nnz)) {
      int64_t pos = indicesPermutationAccessor[j];
      int64_t curr = indicesBufferAccessor[j];
      if (curr == prev) {
        if (values.numel() >
            0) { // if values is an empty tensor, there are no elements to copy
          at::native::cpublas::axpy<scalar_t>(
              blockSize,
              static_cast<scalar_t>(1),
              values_ptr + pos * blockSize,
              1,
              newValues_ptr + i * blockSize,
              1);
        }
      } else {
        ++i;
        for (const auto d : c10::irange(sparse_dim)) {
          newIndicesAccessor[d][i] = indicesAccessor[d][pos];
        }
        if (values.numel() >
            0) { // if values is an empty tensor, there are no elements to copy
          at::native::cpublas::copy<scalar_t>(
              blockSize,
              values_ptr + pos * blockSize,
              1,
              newValues_ptr + i * blockSize,
              1);
        }
      }
      prev = curr;
    }
  });

  dst._coalesced_(true);
  get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

  return dst;
}

DEFINE_DISPATCH(sparse_mask_intersection_out_stub);
DEFINE_DISPATCH(sparse_mask_projection_out_stub);

using OptTensor = std::optional<Tensor>;

static std::tuple<Tensor, Tensor, OptTensor> sparse_mask_like_prepare_sparse_inputs(
    const std::string& method_name,
    const Tensor& t,
    const Tensor& mask) {
  // This is a helper function for operations that implement "sparse_mask"-like
  // functionality, namely, projection of values of one tensor onto the other.
  // These operations mostly rely on COO intersection primitives that heavily
  // exploit coalesced inputs to avoid any syncs and calls to sort. The problem
  // is that these primitives might project first argument onto second one or
  // the other way around depending on which arguments are coalesced and which are
  // larger. This function prepares inputs for `sparse_mask` such that `t` is
  // projected onto `mask` by sorting `t` if uncoalesced and artificially marking it
  // as coalesced all while `mask` is set to uncoalesced.
  // The result of this projectionk is going to be uncoalesced, so it is up to the
  // user to set the corresponding flag correctly with respect to the operations'
  // semantics.

  // We already assume that t.sizes() == mask.sizes()
  TORCH_CHECK(t.sparse_dim() == mask.sparse_dim(),
              method_name, "(): the number of sparse dimensions in `self` ",
              "should match that of the `mask`. ",
              "Got `self.sparse_dim() == ", t.sparse_dim(), "` != ",
              "`mask.sparse_dim() == ", mask.sparse_dim(), "`.");

  const auto wrapped_tensor = [](const Tensor& t,
                                 const OptTensor& indices = std::nullopt,
                                 const OptTensor& values = std::nullopt) -> Tensor {
    auto res = at::empty({0}, t.options());
    auto* res_sparse_impl = get_sparse_impl(res);
    res_sparse_impl->raw_resize_(t.sparse_dim(), t.dense_dim(), t.sizes());
    const auto res_indices = indices.has_value() ? *indices : t._indices();
    const auto res_values = values.has_value() ? *values : t._values();
    res_sparse_impl->set_indices_and_values_unsafe(res_indices, res_values);
    res_sparse_impl->set_nnz_and_narrow(t._nnz());
    res._coalesced_(false);
    return res;
  };

  auto [lhs, lhs_hash_opt, lhs_is_movable] = [&]() -> auto {
    if (t.is_coalesced()) {
      return std::make_tuple(t, static_cast<OptTensor>(std::nullopt), false);
    } else {
      const auto indices_hash = at::sparse::flatten_indices(t._indices(), t.sizes());
      const auto argsort_indices_hash = std::get<1>(indices_hash.sort(0));
      // Probably worth having a dedicated kernel for.
      const auto res_indices = t._indices().index_select(1, argsort_indices_hash);
      const auto res_values = t._values().index_select(0, argsort_indices_hash);
      const auto indices_hash_sorted = indices_hash.index_select(0, argsort_indices_hash);
      // NOTE: res is not necessarily coalesced, but it is sorted.
      // We mark it as "coalesced" to skip sorting in the intersection kernel.
      auto res = wrapped_tensor(t, res_indices, res_values)._coalesced_(true);
      return std::make_tuple(std::move(res), static_cast<OptTensor>(std::move(indices_hash_sorted)), true);
    }
  }();

  const auto rhs = mask.is_coalesced() ? wrapped_tensor(mask) : mask;
  const auto rhs_is_movable = mask.is_coalesced() ? true : false;

  return std::make_tuple(lhs_is_movable ? std::move(lhs) : lhs,
                         rhs_is_movable ? std::move(rhs) : rhs,
                         lhs_hash_opt);
}

SparseTensor sparse_mask(const Tensor& t, const SparseTensor& mask) {
  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  if (t.is_same(mask)) {
    return t;
  }

  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(t.device(), t.scalar_type());
  }

  if (t.layout() == at::kSparse) {
    if (!t._nnz()) {
      auto res = mask.to(t.device(), t.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
      res._values().zero_();
      return res;
    }

    auto res = at::empty({0}, t.options());
    auto [lhs, rhs, lhs_hash_opt] = sparse_mask_like_prepare_sparse_inputs("sparse_mask", t, mask);
    sparse_mask_intersection_out_stub(res.device().type(), res, lhs, rhs, lhs_hash_opt);
    return res._coalesced_(mask.is_coalesced());
  }

  const auto mask_values = mask._values();
  auto mask_template = at::sparse_coo_tensor(
      mask._indices(),
      at::ones({1}, mask_values.options()).expand_as(mask_values),
      mask.sizes())._coalesced_(mask.is_coalesced());
  return t.mul(mask_template).to(t.scalar_type());
}

Tensor sparse_mask_projection(const Tensor& t, const Tensor& mask, bool accumulate_matches) {
  TORCH_INTERNAL_ASSERT(t.is_sparse());
  TORCH_INTERNAL_ASSERT(mask.is_sparse());

  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "_sparse_mask_projection(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  if (!t.numel() || !t._nnz() || !mask._nnz()) {
    auto res = t.clone();
    res._values().zero_();
    return res;
  }

  auto res = at::empty({0}, t.options());
  auto [lhs, rhs, lhs_hash_opt] = sparse_mask_like_prepare_sparse_inputs("_sparse_mask_projection", mask, t);
  sparse_mask_projection_out_stub(res.device().type(), res, lhs, rhs, lhs_hash_opt, accumulate_matches);
  return res._coalesced_(t.is_coalesced());
}

Tensor empty_like_sparse_coo(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");

  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  if (options.layout() == kSparse) {
    auto result = at::empty({0}, options);
    result.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    return result;
  } else {
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  }
}

bool is_pinned_sparse_coo(const Tensor& self, std::optional<Device> device) {
  // Assuming that _indices has the same pin memory state as _values
  return self._values().is_pinned(device);
}

Tensor _pin_memory_sparse_coo(const Tensor& self, std::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  // pinning of sparse tensor is equivalent to cloning indices and
  // values that will not change the sparse tensor invariants. Hence,
  // we can skip checking the sparse tensor invariants for efficiency.
  at::sparse_csr::CheckSparseTensorInvariants _(false);
  TensorOptions options = self.options().pinned_memory(true);
  return at::_sparse_coo_tensor_with_dims_and_tensors(
      self.sparse_dim(),
      self.dense_dim(),
      self.sizes(),
      self._indices().pin_memory(device),
      self._values().pin_memory(device),
      options,
      self.is_coalesced());
}

} // namespace at::native
