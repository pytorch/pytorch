#pragma once

#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <cstdint>

namespace torch {
namespace distributed {
namespace autograd {

// DistAutogradContext which stores information for a single distributed
// autograd pass on a worker.
class TORCH_API DistAutogradContext {
 public:
  explicit DistAutogradContext(int64_t context_id);

  // Retrieves the autograd context id for this context.
  int64_t context_id() const;

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

  std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>> sendFunctions()
      const;

  std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>> recvFunctions()
      const;

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
  const int64_t context_id_;

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

  // Lock to protect concurrent modification of the context.
  mutable std::mutex lock_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
