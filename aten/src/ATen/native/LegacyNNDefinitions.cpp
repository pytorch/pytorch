#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_out(output, self, target, weight, reduction);
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy(self, target, weight, reduction);
}
Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_backward_out(grad_input, grad_output, self, target, weight, reduction);
}

Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  return at::_thnn_binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
}

Tensor & mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_out(output, self, target, reduction);
}

Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss(self, target, reduction);
}

Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_mse_loss_backward(grad_output, self, target, reduction);
}

Tensor & l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_out(output, self, target, reduction);
}

Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss(self, target, reduction);
}

Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
}

Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  return at::_thnn_l1_loss_backward(grad_output, self, target, reduction);
}

}} // namespace at::native
