#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>

namespace at { namespace native {

Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) {
  Tensor is_target = at::empty({0}, self.options());
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

Tensor & nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return std::get<0>(at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

Tensor & nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss2d_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  return std::get<0>(at::nll_loss2d_forward(self, target, weight, reduction, ignore_index));
}

Tensor & log_sigmoid_out(Tensor & output, const Tensor & self) {
  Tensor buffer = at::empty({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(output, buffer, self));
}

Tensor log_sigmoid(const Tensor & self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

Tensor & thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
}

Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  return std::get<0>(at::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding));
}

Tensor & thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

}} // namespace at::native
