#pragma once

#include <torch/tensor.h>

#include <utility>

namespace torch {
namespace nn {
template <typename Derived, typename ReturnType = torch::Tensor>
struct Callable {
  template <typename... Args>
  ReturnType operator()(Args&&... args) {
    return static_cast<Derived*>(this)->forward(std::forward<Args>(args)...);
  }
};
} // namespace nn
} // namespace torch
