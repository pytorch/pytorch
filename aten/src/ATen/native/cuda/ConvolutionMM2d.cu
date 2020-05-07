#include <ATen/ATen.h>

namespace at {
namespace native {

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cuda(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input) {
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    grad_bias.resize_({ weight.size(0) });
    grad_bias.zero_();
  }
  return legacy::cuda::_thnn_conv2d_backward_out(grad_input, grad_weight, grad_bias,
                                                 grad_output, self, weight,
                                                 kernel_size, stride, padding,
                                                 finput, fgrad_input);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  return native::slow_conv2d_backward_out_cuda(grad_input, grad_weight, grad_bias,
                                               grad_output, self, weight,
                                               kernel_size, stride, padding,
                                               finput, fgrad_input);
}

} // namespace native
} // namespace at
