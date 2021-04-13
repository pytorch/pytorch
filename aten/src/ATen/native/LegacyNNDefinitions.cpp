#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>

namespace at { namespace native {


Tensor & thnn_conv_depthwise2d_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  return at::thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  return at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

}} // namespace at::native
