#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/variable_info.h>

namespace torch::autograd {

VariableInfo::VariableInfo(const Variable& var, bool use_zeros_like)
    : layout(var.layout()),
      device(var.device()),
      scalar_type(var.scalar_type()),
      size(var.sym_sizes().vec()),
      requires_grad(var.requires_grad()),
      is_empty(false),
      the_var(
          use_zeros_like ? std::optional<Variable>(var.detach())
                         : std::nullopt) {}

VariableInfo::VariableInfo() : requires_grad(false), is_empty(true) {}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  if (is_empty) {
    // Return undefined tensor.
    return at::Tensor();
  } else if (the_var.has_value()) {
    return at::zeros_like(*the_var);
  } else {
    return at::zeros_symint(
        size, at::TensorOptions(scalar_type).device(device).layout(layout));
  }
}

} // namespace torch::autograd
