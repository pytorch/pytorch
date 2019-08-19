#pragma once

#include <torch/csrc/autograd/function.h>

namespace torch {
namespace distributed {
namespace autograd {

struct TORCH_API SendRpcBackwards : public torch::autograd::Node {
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
