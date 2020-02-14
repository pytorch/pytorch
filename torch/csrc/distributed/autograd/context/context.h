#pragma once

#include <ATen/core/Dict.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
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

  // records the workerID of a node that we sent an RPC to.
  // workerIDs are added here when we attach a send function to this autograd
  // context
  void addKnownWorkerId(const rpc::worker_id_t workerId);

  // Retrieves a set containing the known workerIds for this context
  // These are the different workers that this context has sent RPCs to.
  std::unordered_set<rpc::worker_id_t> getKnownWorkerIds() const;

 private:
  friend class BackwardPassCleanupGuard;
  friend class DistEngine;
  friend class RecvRpcBackward;

  // Record that we would like to accumulate the provided gradient on the given
  // variable.
  void accumulateGrad(
      const torch::autograd::Variable& variable,
      const torch::Tensor& grad,
      size_t num_expected_refs);

  // Retrieve the GraphTask.
  std::shared_ptr<torch::autograd::GraphTask> retrieveGraphTask();

  // Set the appropriate graph task for the backward pass. Can be called only
  // once.
  void setGraphTask(std::shared_ptr<torch::autograd::GraphTask> graphTask);

  // Resets the graph task to ensure we can run another distributed backward
  // pass for the same autograd context.
  void resetGraphTask();

  // Waits for all outstanding RPCs for this context to finish and clears all
  // outstanding rpcs held in this context. This should be called only once.
  std::shared_ptr<rpc::FutureMessage> clearAndWaitForOutstandingRpcsAsync();

  void clearOutstandingRpcs();

  const int64_t contextId_;

  // Set containing known worker IDs, used in cleaning up autograd context.
  // Whenever a sendRpcBackward is attached to the autograd graph for this
  // context, the destination is added here.
  std::unordered_set<rpc::worker_id_t> knownWorkerIds_;

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
  // successfully only if all these futures are done and are successful.
  std::vector<std::shared_ptr<rpc::FutureMessage>> outStandingRpcs_;

  // Lock to protect concurrent modification of the context.
  mutable std::recursive_mutex lock_;
};

using ContextPtr = std::shared_ptr<DistAutogradContext>;

} // namespace autograd
} // namespace distributed
} // namespace torch
