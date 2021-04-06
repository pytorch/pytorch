#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at {
namespace native {

std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out(const Tensor & grad_output,
    const Tensor & self,
    const Tensor & weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor & grad_input,
    Tensor & grad_weight) {
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  return legacy::cuda::_thnn_conv_depthwise2d_backward_out(grad_input, grad_weight,
                                                           grad_output, self, weight,
                                                           kernel_size, stride, padding, dilation);
}

std::tuple<Tensor, Tensor> thnn_conv_depthwise2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    std::array<bool, 2> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  return native::thnn_conv_depthwise2d_backward_out(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      grad_input,
      grad_weight);
}

} // namespace native
} // namespace at
