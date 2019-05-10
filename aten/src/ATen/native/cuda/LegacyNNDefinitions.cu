#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at { namespace native {

Tensor & thnn_conv_depthwise2d_forward_out_cuda(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::cuda::_thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d_forward_cuda(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::cuda::_thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}

std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_cuda(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  return at::legacy::cuda::_thnn_conv_depthwise2d_backward_out(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}

std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward_cuda(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
  return at::legacy::cuda::_thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

}} // namespace at::native
