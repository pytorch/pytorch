#pragma once

#include <torch/csrc/autograd/function.h>

namespace torch {
namespace distributed {
namespace autograd {

// As part of our distributed autograd implementation, whenever we send an RPC
// from one node to another, we add a 'SendRpcBackward' autograd function to the
// autograd graph. This is more or less a placeholder function that is used to
// kickoff the autograd engine on the current worker on the backward pass. The
// edges for this autograd function are the inputs to the RPC method.
//
// During the backward pass, this function is queued for execution in the
// autograd engine which eventually runs the rest of the autograd graph.
struct TORCH_API SendRpcBackward : public torch::autograd::Node {
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
