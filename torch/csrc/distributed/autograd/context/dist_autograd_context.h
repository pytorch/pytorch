#pragma once

#include <ATen/core/Dict.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/rpc/future_message.h>
#include <cstdint>

namespace torch {
namespace distributed {
namespace autograd {

class RecvRpcBackward;

// DistAutogradContext which stores information for a single distributed
// autograd pass on a worker.
class TORCH_API DistAutogradContext {
 public:
  explicit DistAutogradContext(int64_t contextId);

  // Retrieves the autograd context id for this context.
  int64_t contextId() const;

  // Records a 'send' autograd function for this context with the provided
  // message id.
  void addSendFunction(
      const std::shared_ptr<SendRpcBackward>& func,
      int64_t autograd_message_id);

  // Records a 'recv' autograd function for this context with the provided
  // message id.
  void addRecvFunction(
      std::shared_ptr<RecvRpcBackward>& func,
      int64_t autograd_message_id);

  // Given an autograd_message_id, retrieve the appropriate send function.
  std::shared_ptr<SendRpcBackward> retrieveSendFunction(
      int64_t autograd_message_id);

  // Return all send functions for this context.
  std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>> sendFunctions()
      const;

  // Return all recv functions for this context.
  std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>> recvFunctions()
      const;

  // Adds a future message recording an outstanding RPC.
  void addOutstandingRpc(
      const std::shared_ptr<rpc::FutureMessage>& futureMessage);

  // Returns all gradients.
  const c10::Dict<torch::Tensor, torch::Tensor> getGradients() const;

  DistAutogradContext(const DistAutogradContext&) = delete;
  DistAutogradContext& operator=(const DistAutogradContext&) = delete;
  DistAutogradContext(DistAutogradContext&&) = delete;
  DistAutogradContext& operator=(DistAutogradContext&&) = delete;

 private:
  friend class DistEngine;

  // Called once the future for an outstanding RPC has been invoked.
  void outStandingRpcCallback(const rpc::Message& message);

  // Record that we would like to accumulate the provided gradient on the given
  // variable.
  void accumulateGrad(
      const torch::autograd::Variable& variable,
      const torch::Tensor& grad);

  // Retrieve the GraphTask.
  std::shared_ptr<torch::autograd::GraphTask> retrieveGraphTask();

  // Set the appropriate graph task for the backward pass. Can be called only
  // once.
  void setGraphTask(std::shared_ptr<torch::autograd::GraphTask> graphTask);

  // Waits for all outstanding RPCs for this context to finish and clears all
  // outstanding rpcs held in this context. This should be called only once.
  void clearAndWaitForOutstandingRpcs();

  const int64_t contextId_;

  // Map from autograd_message_id to appropriate 'send' autograd function.
  std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>>
      sendAutogradFunctions_;

  // Map from autograd_message_id to appropriate 'recv' autograd function.
  std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
      recvAutogradFunctions_;

  // Gradients accumulated in this context so far. The key is the variable on
  // which the gradient needs to be accumulated and the value is the gradient
  // that needs to be accumulated on that variable..
  c10::Dict<torch::Tensor, torch::Tensor> accumulatedGrads_;

  // The autograd GraphTask for the backward pass on this node for this context.
  std::shared_ptr<torch::autograd::GraphTask> graphTask_;

  // List of futures for RPCs initiated by this node to propagate gradients to
  // other nodes. The distributed autograd engine on this node can return
  // successfully only if all these futures are done and are successfull.
  std::vector<std::shared_ptr<rpc::FutureMessage>> outStandingRpcs_;

  // Lock to protect concurrent modification of the context.
  mutable std::mutex lock_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
