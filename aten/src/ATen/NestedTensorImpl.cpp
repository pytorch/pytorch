#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>

namespace at {
namespace native {

NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor)
    : TensorImpl(
          (c10::DispatchKeySet(DispatchKey::NestedTensor) |
           c10::DispatchKeySet(buffer.is_cuda() ? BackendComponent::CUDABit : BackendComponent::CPUBit)),
          buffer.dtype(),
          buffer.device()),
      buffer_(std::move(buffer)),
      nested_size_tensor_(std::move(nested_size_tensor)) {
  TORCH_WARN_ONCE(
      "The PyTorch API of nested tensors is in prototype stage and will change "
      "in the near future.");
  TORCH_INTERNAL_ASSERT(buffer_.is_cuda() || buffer_.is_cpu(), "NestedTensorImpl buffer must be either CUDA or CPU but got ", buffer_);
  TORCH_INTERNAL_ASSERT(nested_size_tensor_.is_contiguous());
  int64_t size_dim = nested_size_tensor_.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  remove_autograd_key();
  key_set_ =
      key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
  refresh_dim();
  set_sizes_customization_policy(CustomizableMethodPolicy::NotSupported);
}

void NestedTensorImpl::refresh_dim() {
  const auto my_dim = nested_size_tensor_.dim() ? nested_size_tensor_.sizes()[1] + 1 : 1;
  sizes_and_strides_.resize(my_dim);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

const char* NestedTensorImpl::tensorimpl_type_name() const {
  return "NestedTensorImpl";
}
} // namespace native
} // namespace at
