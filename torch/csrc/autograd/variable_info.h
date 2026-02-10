#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

struct TORCH_API VariableInfo {
  explicit VariableInfo();
  explicit VariableInfo(const Variable& var, bool use_zeros_like = false);

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;

  // Store a tiny template tensor (scalar) that preserves the tensor subclass
  // type (e.g., DTensor with its mesh/placements) without holding the full
  // tensor data. This allows creating zeros via new_zeros_symint during
  // backward while using minimal memory.
  void save_template_for_zeros(const Variable& var);

  bool has_template() const {
    return template_var.has_value();
  }

  at::Layout layout = at::Layout::Strided;
  at::Device device = at::kCPU;
  at::ScalarType scalar_type = at::kFloat;
  std::vector<c10::SymInt> size;
  bool requires_grad;
  bool is_empty;
  // For NJT: stored directly since NJT needs zeros_like with exact structure
  std::optional<Variable> the_var;
  // For DTensor: a tiny template (scalar) that preserves DTensor type/spec
  std::optional<Variable> template_var;
};

} // namespace torch::autograd
