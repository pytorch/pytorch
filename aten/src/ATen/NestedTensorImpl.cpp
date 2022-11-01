#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Logging.h>

#include <numeric>
#include <functional>

namespace {
inline void validate_nested_tensor_metadata(
    const at::Tensor& nested_sizes,
    const at::Tensor& nested_strides,
    const std::vector<int64_t>& offsets) {
  TORCH_INTERNAL_ASSERT(nested_sizes.is_contiguous());
  int64_t size_dim = nested_sizes.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  TORCH_INTERNAL_ASSERT(nested_strides.is_contiguous());
  TORCH_INTERNAL_ASSERT(nested_strides.dim() == size_dim);
  TORCH_INTERNAL_ASSERT(nested_sizes.sizes() == nested_strides.sizes());
  TORCH_INTERNAL_ASSERT(
      (size_dim == 0 && (int64_t)offsets.empty()) ||
      (size_dim == 2 && nested_sizes.size(0) == (int64_t)offsets.size()));
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
namespace at {
namespace native {

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
    int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
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
inline at::Tensor construct_nested_stride_tensor(const at::Tensor& sizes) {
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
  const int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
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
   * Create a vector of offsets assuming the nested tensor is contiguous
   *
   * This function iterates over the implicit ntensor outer dimension
   * populating a vector with the num_elements in each implicit tensor.
   * The first element is always 0 and the length of the returned vector
   * is n_tensor.
   *
   * @return A vector of offsets
  */
inline std::vector<int64_t> construct_offsets(const at::Tensor& sizes) {
  // empty `sizes` means empty nested tensor, so return empty strides
  if (sizes.dim() == 0) {
    return std::vector<int64_t>();
  }
  int64_t ntensors = sizes.size(0), orig_dim = sizes.size(1);
  std::vector<int64_t> offsets(ntensors);
  // nesting scalars has easy offsets
  if (orig_dim == 0) {
    std::iota(offsets.begin(), offsets.end(), 0);
    return offsets;
  }
  const int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  offsets[0] = 0;
  for (const auto i : c10::irange(ntensors - 1)) {
    const int64_t row_product = std::accumulate(sizes_ptr, sizes_ptr + orig_dim, 1, std::multiplies<int64_t>());
    offsets[i + 1] = offsets[i] + row_product;
    sizes_ptr += orig_dim;
  }
  return offsets;
}

NestedTensorImpl::NestedTensorImpl(
    Storage storage,
    c10::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    at::Tensor nested_size_tensor,
    at::Tensor nested_stride_tensor,
    std::vector<int64_t>&& offsets)
    : TensorImpl(std::move(storage), key_set, data_type),
      nested_size_tensor_(std::move(nested_size_tensor)),
      nested_stride_tensor_(std::move(nested_stride_tensor)),
      storage_offsets_(std::move(offsets)),
      opt_sizes_(construct_opt_sizes(nested_size_tensor_)) {
  C10_LOG_API_USAGE_ONCE("torch.NestedTensor");
  TORCH_WARN_ONCE(
      "The PyTorch API of nested tensors is in prototype stage and will change "
      "in the near future.");
  auto storage_device = storage_.device();
  TORCH_INTERNAL_ASSERT(
      storage_device.is_cpu() || storage_device.is_cuda(),
      "NestedTensorImpl storage must be either CUDA or CPU but got ",
      storage_device);
  validate_nested_tensor_metadata(nested_size_tensor_, nested_stride_tensor_, storage_offsets_);
  refresh_dim();
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor,
    at::Tensor nested_stride_tensor,
    std::vector<int64_t>&& offsets)
    : NestedTensorImpl(
          buffer.storage(),
          generate_nested_key_set_from_buffer(buffer),
          buffer.dtype(),
          nested_size_tensor,
          nested_stride_tensor,
          std::move(offsets)) {

  TORCH_INTERNAL_ASSERT(
      buffer.dim() == 1,
      "NestedTensorImpl buffer is required to be 1 dimensional but got a buffer with ",
      buffer.dim(),
      " dimensions.");
}

// assume contiguous, `nested_stride_tensor` and `offsets`
// can be infered from `nested_size_tensor`
NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor)
    : NestedTensorImpl(
          buffer,
          nested_size_tensor,
          construct_nested_stride_tensor(nested_size_tensor),
          construct_offsets(nested_size_tensor))
{}

NestedTensorImpl::NestedTensorImpl(
    c10::TensorImpl::ImplType impl_type,
    const at::Tensor& base_tensor,
    at::Tensor nested_size_tensor,
    at::Tensor nested_stride_tensor,
    std::vector<int64_t>&& offsets)
    : TensorImpl(impl_type, Storage(base_tensor.storage()), get_view_key_set(base_tensor), base_tensor.dtype()),
      nested_size_tensor_(std::move(nested_size_tensor)),
      nested_stride_tensor_(std::move(nested_stride_tensor)),
      storage_offsets_(std::move(offsets)),
      opt_sizes_(construct_opt_sizes(nested_size_tensor_)) {
  validate_nested_tensor_metadata(nested_size_tensor_, nested_stride_tensor_, storage_offsets_);
  refresh_dim();
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

void NestedTensorImpl::refresh_dim() {
  const auto my_dim = nested_size_tensor_.dim() ? nested_size_tensor_.sizes()[1] + 1 : 1;
  sizes_and_strides_.resize(my_dim);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

int64_t NestedTensorImpl::dim_custom() const {
  return dim_default();
}

// Currently sizes and strides assume contiguous
int64_t NestedTensorImpl::numel_custom() const {
  if (nested_size_tensor_.dim() == 0) {
    return 0;
  }
  constexpr auto numel_max = std::min(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  const auto nt_dim = nested_size_tensor_.size(1);
  const int64_t* sizes_ptr = nested_size_tensor_.data_ptr<int64_t>();
  uint64_t num_elements{0};

  for (const auto i : c10::irange(nested_size_tensor_.size(0))) {
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


c10::SymInt NestedTensorImpl::sym_numel_custom() const {
  return NestedTensorImpl::numel_custom();
}

bool NestedTensorImpl::is_contiguous_custom(MemoryFormat) const {
  return nested_tensor_impl_is_contiguous(this);
}
IntArrayRef NestedTensorImpl::sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
}
c10::SymIntArrayRef NestedTensorImpl::sym_sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
}

c10::SymIntArrayRef NestedTensorImpl::sym_strides_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
}

IntArrayRef NestedTensorImpl::strides_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
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
    auto r = (*pyobj_interpreter_.load(std::memory_order_acquire))->detach(this);
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
      nested_size_tensor_,
      nested_stride_tensor_,
      std::vector<int64_t>(storage_offsets_));

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

} // namespace native
} // namespace at
