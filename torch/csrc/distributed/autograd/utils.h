#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backwards.h>

namespace torch {
namespace distributed {
namespace autograd {

// This method is used to attach the 'send' autograd function to the autograd
// graph when we use RPC. This method creates a new 'send' autograd function
// and attaches the provided IValues as next_edges to the 'send' function. Only
// IValues of type 'Tensor' are attached.
// TODO: Support Tensors which might be nested inside other IValue types
//       (ex: List, Tuple)
//
// Returns a shared_ptr to the autograd function, so that we can hold a
// reference to it.
std::shared_ptr<SendRpcBackwards> addSendRpcBackward(
    at::ArrayRef<c10::IValue> ivalues);

} // namespace autograd
} // namespace distributed
} // namespace torch
