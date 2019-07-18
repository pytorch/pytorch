#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace native {

Tensor quantized_view(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }

  auto stride = at::detail::computeStride(self.sizes(),
                                           self.strides(),
                                           inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  auto self_ = self.clone();
  self_.set_(self.storage(), self.storage_offset(), inferred_size,
             stride_value);
  return self_;
}

} // namespace native
} // namespace at
