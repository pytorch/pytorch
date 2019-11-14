#pragma once

#include <torch/types.h>

namespace torch {

inline at::Tensor isfinite(const at::Tensor& tensor) {
  return at::isfinite(tensor);
}

} // namespace torch
