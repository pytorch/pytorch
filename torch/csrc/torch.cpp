#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
at::Tensor as_variable(at::Tensor tensor, bool requires_grad) {
  return autograd::make_variable(tensor, requires_grad);
}
} // namespace torch
