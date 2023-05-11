#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace native {
Tensor view_as_nested(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_nested() && other.is_nested(),
      "We currently can only view non-nested Tensors as nested Tensors. ",
      "Expected self to not be nested and other to be nested but got ",
      self.is_nested(),
      " and ",
      other.is_nested(),
      " instead.");
  TORCH_CHECK(self.is_contiguous(), "We currently only support contiguous Tensors for self when other is nested.");
  TORCH_CHECK(other.is_contiguous(), "We currently can only view as contiguous nested Tensors.");
  TORCH_CHECK(self.dim() == 1, "We currently only support 1-dim Tensors for self when other is nested. Got ", self.dim(), " instead.");
  TORCH_CHECK(self.layout() == at::kStrided, "We currently only support strided Tensors for self when other is nested. Got ", self.layout(), " instead.");
  TORCH_CHECK(self.numel() == other.numel(), "Expected the number of elements of self and other to match, but got ", self.numel(), " and ", other.numel(), " instead.");
  auto other_impl = get_nested_tensor_impl(other);
  return at::detail::make_tensor<NestedTensorImpl>(
    c10::TensorImpl::VIEW,
    self,
    other_impl->get_nested_sizes().clone());
}
} // namespace native
} // namespace at
