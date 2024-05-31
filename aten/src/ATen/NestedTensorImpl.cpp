#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Logging.h>

#include <numeric>
#include <functional>
#include <utility>

namespace {
inline void validate_nested_tensor_metadata(
    const at::Tensor& nested_sizes,
    const at::Tensor& nested_strides,
    const at::Tensor& offsets) {
  TORCH_INTERNAL_ASSERT(nested_sizes.is_contiguous());
  int64_t size_dim = nested_sizes.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  TORCH_INTERNAL_ASSERT(nested_strides.is_contiguous());
  TORCH_INTERNAL_ASSERT(nested_strides.dim() == size_dim);
  TORCH_INTERNAL_ASSERT(nested_sizes.sizes() == nested_strides.sizes());
  TORCH_INTERNAL_ASSERT(
      (size_dim == 0 && offsets.size(0) == 0) ||
      (size_dim == 2 && nested_sizes.size(0) == offsets.size(0)));
}

/**
 * Generates a nested key_set from a non-nested tensor.
 *
 * When creating a nested tensor from a non-nested tensor
 * We want to maintain the same keyset as the buffer but
 * swap non nested keys for nested ones
 *
 * @return Appropriate key set for nested tensor
 */
inline c10::DispatchKeySet generate_nested_key_set_from_buffer(
    const at::Tensor& buffer) {
  auto nested_key_set = buffer.key_set();
  const bool has_autograd = nested_key_set.has_any(c10::autograd_dispatch_keyset);
  // Remove non_nested tensor specific keys
  nested_key_set = nested_key_set -
      c10::DispatchKeySet{c10::DispatchKey::Dense, c10::DispatchKey::Autograd};

  // Add nested tensor specific keys
  nested_key_set =
      nested_key_set | c10::DispatchKeySet{c10::DispatchKey::NestedTensor};
  nested_key_set =
      has_autograd ? nested_key_set | c10::autograd_nested : nested_key_set;
  return nested_key_set;
}

/**
 * Generates a the correct view keyset.
 *
 * When creating a nested tensor view of base
 * The appropriate keyset will be dependent on the nested
 * status of the base
 *
 * @return Appropriate key set for nested tensor
 */
c10::DispatchKeySet get_view_key_set(const at::Tensor& base) {
  return base.is_nested() ? base.key_set()
                          : generate_nested_key_set_from_buffer(base);
}

} // namespace

namespace at::native {

inline std::vector<int64_t> construct_opt_sizes(const at::Tensor& sizes) {
  // torch.tensor([]) is considered to have `dim() = 1` and `size(0) = 0`
  // torch.nested_tensor([]) should also has `dim() = 1` and `size(0) = 0`
  if (sizes.dim() == 0) {
    return std::vector<int64_t>({0});
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  std::vector<int64_t> result(1, sizes.sizes()[0]);
  if (sizes.dim() > 0) {
    size_t nested_dim = result.size();
    const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
    result.resize(nested_dim + sizes.sizes()[1]);
    int64_t sizes_size_0 = sizes.sizes()[0];
    int64_t sizes_size_1 = sizes.sizes()[1];
    for (const auto i : c10::irange(sizes_size_1)) {
      result[nested_dim + i] = sizes_ptr[i];
    }
    for (const auto j : c10::irange(sizes_size_1)) {
      for (const auto i : c10::irange(sizes_size_0)) {
        if (result[nested_dim + j] &&
            (result[nested_dim + j] != sizes_ptr[i * sizes.size(1) + j])) {
          result[nested_dim + j] = -1;
        }
      }
    }
  }
  return result;
}

// assume contiguous, we can construct stride from size
at::Tensor construct_nested_strides(const at::Tensor& sizes) {
  // empty `sizes` means empty nested tensor, so return empty strides
  if (sizes.dim() == 0) {
    return sizes;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  int64_t orig_dim = sizes.size(1);
  // `sizes`.sizes() = ntensors x 0 means empty but shaped `sizes`
  // in this case strides is also empty but shaped
  if (orig_dim == 0) {
    return sizes;
  }
  at::Tensor strides = sizes.new_empty(sizes.sizes());
  const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
  int64_t* strides_ptr = strides.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes.size(0); i++) {
    strides_ptr[orig_dim - 1] = 1;
    int64_t product = sizes_ptr[orig_dim - 1];
    for (int64_t j = orig_dim - 2; j >= 0; j--) {
      strides_ptr[j] = product;
      product *= sizes_ptr[j];
    }
    sizes_ptr += orig_dim;
    strides_ptr += orig_dim;
  }
  return strides;
}

/**
   * Create a tensor of offsets assuming the nested tensor is contiguous
   *
   * This function iterates over the implicit ntensor outer dimension
   * populating a tensor with the num_elements in each implicit tensor.
   * The first element is always 0 and the length of the returned tensor
   * is n_tensor.
   *
   * @return A tensor of offsets
  */
at::Tensor construct_offsets(const at::Tensor& sizes) {
  // empty `sizes` means empty nested tensor, so return empty strides
  if (sizes.dim() == 0) {
    return at::empty({0}, sizes.options().dtype(kLong));
  }
  int64_t ntensors = sizes.size(0), orig_dim = sizes.size(1);
  auto offsets = at::empty({ntensors}, sizes.options());
  int64_t *offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  // nesting scalars has easy offsets
  if (orig_dim == 0) {
    std::iota(offsets_ptr, offsets_ptr + ntensors, 0);
    return offsets;
  }
  const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  for (const auto i : c10::irange(ntensors - 1)) {
    const int64_t row_product = std::accumulate(sizes_ptr, sizes_ptr + orig_dim, 1, std::multiplies());
    offsets_ptr[i + 1] = offsets_ptr[i] + row_product;
    sizes_ptr += orig_dim;
  }
  return offsets;
}

NestedTensorImpl::NestedTensorImpl(
    Storage storage,
    c10::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    : TensorImpl(std::move(storage), key_set, data_type),
      nested_sizes_(std::move(nested_sizes)),
      nested_strides_(std::move(nested_strides)),
      storage_offsets_(std::move(storage_offsets)),
      opt_sizes_(c10::nullopt) {
  C10_LOG_API_USAGE_ONCE("torch.NestedTensor");
  TORCH_WARN_ONCE(
      "The PyTorch API of nested tensors is in prototype stage and will change "
      "in the near future.");
  auto storage_device = storage_.device();
  TORCH_INTERNAL_ASSERT(
      storage_device.is_cpu() || storage_device.is_cuda() || storage_device.is_xpu() || storage_device.is_privateuseone(),
      "NestedTensorImpl storage must be either CUDA, CPU, XPU or ", get_privateuse1_backend(), " but got ",
      storage_device);
  validate_nested_tensor_metadata(nested_sizes_, nested_strides_, storage_offsets_);
  refresh_dim();
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

NestedTensorImpl::NestedTensorImpl(
    const at::Tensor& buffer,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    : NestedTensorImpl(
          buffer.storage(),
          generate_nested_key_set_from_buffer(buffer),
          buffer.dtype(),
          std::move(nested_sizes),
          std::move(nested_strides),
          std::move(storage_offsets)) {

  TORCH_INTERNAL_ASSERT(
      buffer.dim() == 1,
      "NestedTensorImpl buffer is required to be 1 dimensional but got a buffer with ",
      buffer.dim(),
      " dimensions.");
}

// assume contiguous, `nested_strides` and `offsets`
// can be infered from `nested_sizes`
NestedTensorImpl::NestedTensorImpl(
    const at::Tensor& buffer,
    const at::Tensor& nested_sizes)
    : NestedTensorImpl(
          buffer,
          nested_sizes,
          construct_nested_strides(nested_sizes),
          construct_offsets(nested_sizes))
{}

NestedTensorImpl::NestedTensorImpl(
    c10::TensorImpl::ImplType impl_type,
    const at::Tensor& base_tensor,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    : TensorImpl(impl_type, Storage(base_tensor.storage()), get_view_key_set(base_tensor), base_tensor.dtype()),
      nested_sizes_(std::move(nested_sizes)),
      nested_strides_(std::move(nested_strides)),
      storage_offsets_(std::move(storage_offsets)),
      opt_sizes_(c10::nullopt) {
  validate_nested_tensor_metadata(nested_sizes_, nested_strides_, storage_offsets_);
  refresh_dim();
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

std::optional<int64_t> NestedTensorImpl::opt_size(int64_t d) const {
  if (C10_UNLIKELY(!opt_sizes_.has_value())) {
    // Cache the metadata to avoid recomputing it each time.
    opt_sizes_ = c10::make_optional(construct_opt_sizes(nested_sizes_));
  }
  d = at::maybe_wrap_dim(d, dim(), false);
  if ((*opt_sizes_)[d] == -1) {
    return c10::nullopt;
  }
  return (*opt_sizes_)[d];
}

void NestedTensorImpl::refresh_dim() {
  const auto my_dim = nested_sizes_.dim() ? nested_sizes_.sizes()[1] + 1 : 1;
  sizes_and_strides_.resize(my_dim);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

int64_t NestedTensorImpl::dim_custom() const {
  return dim_default();
}

// Currently sizes and strides assume contiguous
int64_t NestedTensorImpl::numel_custom() const {
  if (nested_sizes_.dim() == 0) {
    return 0;
  }
  return get_numel_from_nested_size_tensor(nested_sizes_);
}


c10::SymInt NestedTensorImpl::sym_numel_custom() const {
  return NestedTensorImpl::numel_custom();
}

bool NestedTensorImpl::is_contiguous_custom(MemoryFormat) const {
  return nested_tensor_impl_is_contiguous(this);
}
IntArrayRef NestedTensorImpl::sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue.");
}
c10::SymIntArrayRef NestedTensorImpl::sym_sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue.");
}

c10::SymIntArrayRef NestedTensorImpl::sym_strides_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue.");
}

IntArrayRef NestedTensorImpl::strides_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue.");
}

const char* NestedTensorImpl::tensorimpl_type_name() const {
  return "NestedTensorImpl";
}


template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  if (key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    auto r = pyobj_slot_.load_pyobj_interpreter()->detach(this);
    if (r) {
      r->set_version_counter(std::forward<VariableVersion>(version_counter));
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
    // otherwise just copy the TensorImpl and not the PyObject.  Since
    // the interpreter is dead no one can call us out on it
  }
  auto impl = c10::make_intrusive<NestedTensorImpl>(
      storage_,
      key_set_,
      data_type_,
      nested_sizes_,
      nested_strides_,
      storage_offsets_);

      copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/std::forward<VariableVersion>(version_counter),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

int64_t get_numel_from_nested_size_tensor(const at::Tensor& tensor) {
  constexpr auto numel_max = std::min(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  const int64_t* sizes_ptr = tensor.const_data_ptr<int64_t>();
  const auto nt_dim = tensor.size(1);
  uint64_t num_elements{0};

  for (const auto i : c10::irange(tensor.size(0))) {
    uint64_t n = 1;
    const auto start{sizes_ptr + i * nt_dim};
    const auto end{start + nt_dim};
    bool overflows = c10::safe_multiplies_u64(start, end, &n);
    num_elements += n;
    overflows |= (num_elements > numel_max);
    TORCH_CHECK(!overflows, "numel: integer multiplication overflow");
  }
  return static_cast<int64_t>(num_elements);
}

} // namespace at::native
