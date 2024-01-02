#pragma once
#include <ATen/core/Tensor.h>

namespace torch::jit {

// a wrapper to mark places where we expect all the at::Tensors to be
// variables
struct variable_tensor_list : public std::vector<at::Tensor> {
  variable_tensor_list() = default;
  template <class InputIt>
  variable_tensor_list(InputIt first, InputIt last)
      : std::vector<at::Tensor>(first, last) {}
  explicit variable_tensor_list(std::vector<at::Tensor>&& tensor)
      : std::vector<at::Tensor>(std::move(tensor)) {}
};

} // namespace torch::jit
