#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
autograd::VariableType getType(at::Backend backend, at::ScalarType type) {
  return autograd::VariableType(
      &at::globalContext(), &at::getType(backend, type));
}

autograd::VariableType CPU(at::ScalarType type) {
  return torch::getType(at::kCPU, type);
}

autograd::VariableType CUDA(at::ScalarType type) {
  return torch::getType(at::kCUDA, type);
}

void set_requires_grad(at::Tensor& tensor, bool requires_grad) noexcept {
  autograd::as_variable_ref(tensor, /*check=*/true)
      .set_requires_grad(requires_grad);
}

bool requires_grad(const at::Tensor& tensor) noexcept {
  return autograd::as_variable_ref(tensor, /*check=*/true).requires_grad();
}
} // namespace torch
