#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace torch {
namespace distributed {
namespace autograd {

// Used to propagate gradients from one node to another during a distributed
// backwards pass. This RPC call is invoked when we hit a `recv` autograd
// function during backward pass execution.
class TORCH_API PropagateGradientsReq : public rpc::RpcCommandBase {
 public:
  PropagateGradientsReq(
      const AutogradMetadata& autogradMetadata,
      std::vector<torch::autograd::Variable> grads);

  const AutogradMetadata& getAutogradMetadata();

  const std::vector<torch::autograd::Variable>& getGrads();

  // Serialization and deserialization methods.
  rpc::Message toMessage() && override;
  static std::unique_ptr<PropagateGradientsReq> fromMessage(
      const rpc::Message& message);

 private:
  AutogradMetadata autogradMetadata_;
  std::vector<torch::autograd::Variable> grads_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
