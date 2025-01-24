#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch::distributed::autograd {

// Forward declarations.
class DistAutogradContext;

// As part of our distributed autograd implementation, whenever we receive an
// RPC from a node, we add a 'RecvRpcBackward' autograd function to the
// autograd graph. This is more or less a placeholder function that is used to
// pass gradients to the remote host during the backward pass. The inputs to the
// RPC function are the inputs to this autograd function.
class TORCH_API RecvRpcBackward : public torch::autograd::Node {
 public:
  explicit RecvRpcBackward(
      const AutogradMetadata& autogradMetadata,
      const std::shared_ptr<DistAutogradContext>& autogradContext,
      rpc::worker_id_t fromWorkerId,
      rpc::DeviceMap deviceMap);

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;

 private:
  const AutogradMetadata autogradMetadata_;

  // Hold a weak reference to the autograd context to avoid circular
  // dependencies with the context (since it holds a reference to
  // RecvRpcBackward).
  std::weak_ptr<DistAutogradContext> autogradContext_;

  // The worker id from which the RPC was received. During the backward pass,
  // we need to propagate the gradients to this workerId.
  rpc::worker_id_t fromWorkerId_;

  // Device mapping for tensors sent over RPC.
  const rpc::DeviceMap deviceMap_;
};

} // namespace torch::distributed::autograd
