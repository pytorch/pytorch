#include <ATen/native/UnaryInvolutionFallback.h>

#include <ATen/core/Tensor.h>

namespace at::native::detail {

UnaryInvolutionFallback::~UnaryInvolutionFallback() = default;

auto UnaryInvolutionFallback::transform(Tensor const& tensor) const -> Tensor {
  return tensor.clone();
}

auto UnaryInvolutionFallback::untransform(Tensor& output, Tensor const& result) const -> void {
  output.copy_(result);
}

} // namespace at::native::detail
