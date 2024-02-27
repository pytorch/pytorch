#include <torch/csrc/autograd/variable_info.h>
#include <torch/csrc/autograd/variable.h>
#include "ATen/ops/zeros.h"

namespace torch {
namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
    : layout(var.layout()),
      device(var.device()),
      scalar_type(var.scalar_type()),
      size(var.sym_sizes().vec()),
      requires_grad(var.requires_grad()),
      is_empty(false) {}

VariableInfo::VariableInfo() : requires_grad(false), is_empty(true) {}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  if (is_empty) {
    // Return undefined tensor.
    return at::Tensor();
  } else {
    return at::zeros_symint(
        size, at::TensorOptions(scalar_type).device(device).layout(layout));
  }
}

}
}
