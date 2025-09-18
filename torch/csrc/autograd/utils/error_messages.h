#pragma once

#include <sstream>

namespace torch::autograd::utils {

inline std::string requires_grad_leaf_error() {
  std::ostringstream oss;
  oss << "If you want to use a computed variable in a subgraph "
         "that doesn't require differentiation use "
         "var_no_grad = var.detach().";
  return oss.str();
}

} // namespace torch::autograd::utils
