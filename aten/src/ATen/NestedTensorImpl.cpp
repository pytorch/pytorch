#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>

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

// assume contiguous, we can construct offsets from size
inline std::vector<int64_t> construct_offsets(const at::Tensor& sizes) {
  // empty `sizes` means empty nested tensor, so return empty strides
  if (sizes.dim() == 0) {
    return std::vector<int64_t>();
  }
  int64_t ntensors = sizes.size(0),
      orig_dim = sizes.size(1);
  std::vector<int64_t> offsets(ntensors);
  // nesting scalars has easy offsets
  if (orig_dim == 0) {
    std::iota(offsets.begin(), offsets.end(), 0);
    return offsets;
  }
  const int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t i = 0; i < ntensors - 1; i++) {
    int64_t row_product = sizes_ptr[0];
    for (int64_t j = 1; j < orig_dim; j++) {
      row_product *= sizes_ptr[j];
    }
    offsets[i + 1] = offsets[i] + row_product;
    sizes_ptr += orig_dim;
  }
  return offsets;
}

// [Note: Nested Tensor Autograd] The Nested Tensor key is a functionality
// key and therefore getAutogradRelatedKeySetFromBackend will return the
// wrong autograd key. For this specific impl we make sure to register the
// correct Autograd key which is AutogradNestedTensor
c10::DispatchKeySet generate_nested_key_set(at::Tensor buffer) {
  c10::DispatchKeySet key_set =
      (c10::DispatchKeySet(DispatchKey::NestedTensor) |
       c10::DispatchKeySet(
           buffer.is_cuda() ? BackendComponent::CUDABit
                            : BackendComponent::CPUBit));

  // Add AutogradNestedTensor specific keys
  key_set = key_set | inplace_or_view_ks | autograd_nested;
  return key_set;
}

NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor,
    at::Tensor nested_stride_tensor,
    const std::vector<int64_t>& offsets)
    : TensorImpl(
          generate_nested_key_set(buffer),
          buffer.dtype(),
          buffer.device()),
      buffer_(std::move(buffer)),
      nested_size_tensor_(std::move(nested_size_tensor)),
      nested_stride_tensor_(std::move(nested_stride_tensor)),
      offsets_(offsets),
      opt_sizes_(construct_opt_sizes(nested_size_tensor_))
{
  TORCH_WARN_ONCE(
      "The PyTorch API of nested tensors is in prototype stage and will change "
      "in the near future.");
  TORCH_INTERNAL_ASSERT(buffer_.is_cuda() || buffer_.is_cpu(), "NestedTensorImpl buffer must be either CUDA or CPU but got ", buffer_);
  TORCH_INTERNAL_ASSERT(nested_size_tensor_.is_contiguous());
  int64_t size_dim = nested_size_tensor_.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  TORCH_INTERNAL_ASSERT(nested_stride_tensor_.is_contiguous());
  TORCH_INTERNAL_ASSERT(nested_stride_tensor_.dim() == size_dim);
  TORCH_INTERNAL_ASSERT(nested_stride_tensor_.sizes() == nested_size_tensor_.sizes());
  TORCH_INTERNAL_ASSERT((size_dim == 0 && (int64_t)offsets_.empty())
      || (size_dim == 2 && nested_size_tensor_.size(0) == (int64_t)offsets_.size()));
  refresh_dim();
  set_sizes_strides_policy(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
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

bool NestedTensorImpl::is_contiguous_custom(MemoryFormat) const {
  TORCH_CHECK(false, "is_contiguous is disabled.");
}
IntArrayRef NestedTensorImpl::sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
}
c10::SymIntArrayRef NestedTensorImpl::sym_sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
}

c10::SymIntArrayRef NestedTensorImpl::sym_sizes() const {
  return sym_sizes_custom();
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
    auto r = pyobj_interpreter_.load(std::memory_order_acquire)->detach(this);
    if (r) {
      r->set_version_counter(std::forward<VariableVersion>(version_counter));
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
    // otherwise just copy the TensorImpl and not the PyObject.  Since
    // the interpreter is dead no one can call us out on it
  }
  auto impl = c10::make_intrusive<NestedTensorImpl>(this -> buffer_, this -> nested_size_tensor_);
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
