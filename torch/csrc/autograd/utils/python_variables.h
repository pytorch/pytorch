#pragma once

#include <ATen/ATen.h>
#include "torch/csrc/autograd/python_variable.h"

namespace torch { namespace autograd { namespace utils {

inline at::Tensor set_requires_grad(at::Tensor self, bool requires_grad) {
  if (requires_grad && !at::isFloatingType(self.type().scalarType())) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  as_variable_ref(self).set_requires_grad(requires_grad);
  return self;
}

}}} // namespace torch::autograd::utils
