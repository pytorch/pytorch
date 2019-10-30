#pragma once

#include <torch/types.h>

namespace torch {

inline at::Tensor isfinite(const at::Tensor& tensor) {
  if (!tensor.is_floating_point()) {
    return torch::ones_like(tensor, torch::kBool);
  }
  return (tensor == tensor) * (tensor.abs() != torch::full_like(tensor, std::numeric_limits<double>::infinity()));
}

} // namespace torch
