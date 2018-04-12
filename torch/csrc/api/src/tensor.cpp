#include <torch/tensor.h>

#include <torch/csrc/autograd/generated/VariableType.h>

#include <ATen/ATen.h>

namespace torch {
at::Type& getType(at::Backend backend, at::ScalarType type) {
  return *autograd::VariableType::getType(at::getType(backend, type));
}

at::Type& CPU(at::ScalarType type) {
  return torch::getType(at::kCPU, type);
}

at::Type& CUDA(at::ScalarType type) {
  return torch::getType(at::kCUDA, type);
}
} // namespace torch
