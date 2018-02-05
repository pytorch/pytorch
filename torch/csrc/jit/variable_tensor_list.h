#pragma once
#include "ATen/ATen.h"

namespace torch { namespace jit {

// a wrapper to mark places where we expect all the at::Tensors to be
// variables
struct variable_tensor_list : public std::vector<at::Tensor> {
  variable_tensor_list() {}
  explicit variable_tensor_list(std::vector<at::Tensor> && tensor)
  : std::vector<at::Tensor>(std::move(tensor)) {}
};

}}
