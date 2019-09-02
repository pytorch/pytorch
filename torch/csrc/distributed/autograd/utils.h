#pragma once

#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace autograd {

// This method is used to attach the 'send' autograd function to the autograd
// graph when we use RPC. This method creates a new 'send' autograd function
// and attaches the provided tensors as next_edges to the 'send' function.
//
// Returns a shared_ptr to the autograd function, so that we can hold a
// reference to it.
TORCH_API std::shared_ptr<SendRpcBackward> addSendRpcBackward(
    const std::vector<torch::Tensor>& tensors);

} // namespace autograd
} // namespace distributed
} // namespace torch
