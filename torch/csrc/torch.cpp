#include <torch/csrc/variable_tensor_functions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
at::Type& getType(at::Backend backend, at::ScalarType type) {
  return *autograd::VariableType::getType(at::getType(backend, type));
}

at::Type& CPU(at::ScalarType type) {
  return torch::getType(at::Backend::CPU, type);
}

at::Type& CUDA(at::ScalarType type) {
  return torch::getType(at::Backend::CUDA, type);
}

at::Tensor toTensor(const at::Scalar& scalar) {
  return autograd::make_variable(scalar.toTensor());
}

void set_requires_grad(at::Tensor& tensor, bool requires_grad) noexcept {
  autograd::as_variable_ref(tensor).set_requires_grad(requires_grad);
}

bool requires_grad(const at::Tensor& tensor) noexcept {
  return autograd::as_variable_ref(tensor).requires_grad();
}
} // namespace torch
