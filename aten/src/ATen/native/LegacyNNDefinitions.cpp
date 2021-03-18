#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>

namespace at { namespace native {


Tensor & thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  return at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

}} // namespace at::native
