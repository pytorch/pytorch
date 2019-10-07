#pragma once

#include <torch/csrc/autograd/function.h>

namespace torch {
namespace distributed {
namespace autograd {

// As part of our distributed autograd implementation, whenever we receive an
// RPC from a node, we add a 'RecvRpcBackward' autograd function to the
// autograd graph. This is more or less a placeholder function that is used to
// pass gradients to the remote host during the backward pass. The inputs to the
// RPC function are the inputs to this autograd function.
struct TORCH_API RecvRpcBackward : public torch::autograd::Node {
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
