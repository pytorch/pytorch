#pragma once

#include <torch/csrc/autograd/custom_function.h>

namespace torch::lazy {

struct MaxPool3dAutogradFunctionTS
    : public torch::autograd::Function<MaxPool3dAutogradFunctionTS> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

} // namespace torch::lazy
