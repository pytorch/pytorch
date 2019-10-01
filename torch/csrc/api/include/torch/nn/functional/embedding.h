#pragma once

namespace torch {
namespace nn {
namespace functional {

inline Tensor one_hot(const Tensor& x, int64_t num_classes = -1) {
  return torch::one_hot(x, num_classes);
}
} // namespace functional
} // namespace nn
} // namespace torch
