#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

struct TORCH_API VariableInfo {
  explicit VariableInfo();
  explicit VariableInfo(const Variable& var);

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;

  at::Layout layout = at::Layout::Strided;
  at::Device device = at::kCPU;
  at::ScalarType scalar_type = at::kFloat;
  std::vector<c10::SymInt> size;
  bool requires_grad;
  bool is_empty;
};

} // namespace torch::autograd
