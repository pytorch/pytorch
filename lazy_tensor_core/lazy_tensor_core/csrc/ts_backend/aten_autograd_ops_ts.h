#pragma once

#include <torch/script.h>

namespace torch_lazy_tensors {

namespace aten_autograd_ops_ts {

struct MaxPool3dAutogradFunctionTS
    : public torch::autograd::Function<MaxPool3dAutogradFunctionTS> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor self,
                               torch::IntArrayRef kernel_size,
                               torch::IntArrayRef stride,
                               torch::IntArrayRef padding,
                               torch::IntArrayRef dilation, bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

}  // namespace aten_autograd_ops_ts
}  // namespace torch_lazy_tensors
