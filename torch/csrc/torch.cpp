#include <torch/csrc/variable_tensor_functions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
at::TypeExtendedInterface& getVariableType(at::Backend backend, at::ScalarType type) {
  return *autograd::VariableType::getVariableTypeFromBaseType(at::getNonVariableType(backend, type));
}

at::TypeExtendedInterface& CPU(at::ScalarType type) {
  return torch::getVariableType(at::Backend::CPU, type);
}

at::TypeExtendedInterface& CUDA(at::ScalarType type) {
  return torch::getVariableType(at::Backend::CUDA, type);
}

at::Tensor toTensor(const at::Scalar& scalar) {
  return autograd::make_variable(scalar_to_tensor(scalar));
}

void set_requires_grad(at::Tensor& tensor, bool requires_grad) noexcept {
  autograd::as_variable_ref(tensor).set_requires_grad(requires_grad);
}

bool requires_grad(const at::Tensor& tensor) noexcept {
  return autograd::as_variable_ref(tensor).requires_grad();
}
} // namespace torch
