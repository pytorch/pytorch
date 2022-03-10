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
          // TODO: This doesn't properly report is_cpu/is_cuda for NestedTensor.
          // The intended resolution is that once #72827 lands we will be able to
          // allocate separate dispatch keys for CPUNestedTensor (and any other
          // hypothetical device backends for NestedTensor); then we will be
          // able to derive this directly.  If you need this to work before then,
          // make sure you add CPU to this dispatch key set
          c10::DispatchKeySet({DispatchKey::NestedTensor}),
          buffer.dtype(),
          buffer.device()),
      buffer_(std::move(buffer)),
      nested_size_tensor_(std::move(nested_size_tensor)) {
  TORCH_INTERNAL_ASSERT(nested_size_tensor_.is_contiguous());
  int64_t size_dim = nested_size_tensor_.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  remove_autograd_key();
  key_set_ =
      key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
  refresh_dim();
}

void NestedTensorImpl::refresh_dim() {
  const auto my_dim = nested_size_tensor_.dim() ? nested_size_tensor_.sizes()[1] + 1 : 1;
  sizes_and_strides_.resize(my_dim);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

} // namespace native
} // namespace at
