#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

struct TORCH_API VariableInfo {
  explicit VariableInfo();
  explicit VariableInfo(const Variable& var, bool use_zeros_like = false);

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;

  at::Layout layout = at::Layout::Strided;
  at::Device device = at::kCPU;
  at::ScalarType scalar_type = at::kFloat;
  std::vector<c10::SymInt> size;
  bool requires_grad;
  bool is_empty;
  // needed for e.g. NJTs since they only support zeros_like()
  std::optional<Variable> the_var;
};

} // namespace torch::autograd
