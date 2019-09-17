#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <c10/util/Exception.h>

namespace torch {
namespace distributed {
namespace autograd {

DistAutogradContext::DistAutogradContext(int64_t context_id)
    : context_id_(context_id) {}

int64_t DistAutogradContext::context_id() const {
  return context_id_;
}

std::vector<rpc::WorkerId> DistAutogradContext::getKnownWorkerIds() const {
  return knownWorkerIDs_;
};

void DistAutogradContext::addKnownWorkerID(const rpc::WorkerId& workerId) {
  std::lock_guard<std::mutex> guard(lock_);
  if (std::find(knownWorkerIDs_.begin(), knownWorkerIDs_.end(), workerId) ==
      knownWorkerIDs_.end()) {
    knownWorkerIDs_.push_back(workerId);
  }
}

void DistAutogradContext::addSendFunction(
    const std::shared_ptr<SendRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      sendAutogradFunctions_.find(autograd_message_id) ==
      sendAutogradFunctions_.end());
  sendAutogradFunctions_.emplace(autograd_message_id, func);
}
void DistAutogradContext::addRecvFunction(
    std::shared_ptr<RecvRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      recvAutogradFunctions_.find(autograd_message_id) ==
      recvAutogradFunctions_.end());
  recvAutogradFunctions_.emplace(autograd_message_id, func);
}

std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>>
DistAutogradContext::sendFunctions() const {
  return sendAutogradFunctions_;
}

std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
DistAutogradContext::recvFunctions() const {
  return recvAutogradFunctions_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
