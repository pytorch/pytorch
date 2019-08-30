#pragma once

#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <cstdint>

namespace torch {
namespace distributed {
namespace autograd {

// DistAutogradContext which stores information for a single distributed
// autograd pass on a worker.
class DistAutogradContext {
 public:
  explicit DistAutogradContext(int64_t context_id);

  // Retrieves the autograd context id for this context.
  int64_t context_id() const;

  // Records a 'send' autograd function for this context.
  void addSendFunction(std::shared_ptr<SendRpcBackward> func);

  std::vector<std::shared_ptr<SendRpcBackward>> sendFunctions() const;

  DistAutogradContext(const DistAutogradContext&) = delete;
  DistAutogradContext& operator=(const DistAutogradContext&) = delete;
  DistAutogradContext(DistAutogradContext&&) = delete;
  DistAutogradContext& operator=(DistAutogradContext&&) = delete;

 private:
  const int64_t context_id_;

  std::vector<std::shared_ptr<SendRpcBackward>> sendAutogradFunctions_;

  // Lock to protect concurrent modification of the context.
  mutable std::mutex lock_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
