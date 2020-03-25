#include <functional>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;

DistAutogradContext::DistAutogradContext(int64_t contextId)
    : contextId_(contextId) {}

int64_t DistAutogradContext::contextId() const {
  return contextId_;
}

std::unordered_set<rpc::worker_id_t> DistAutogradContext::getKnownWorkerIds()
    const {
  std::lock_guard<std::mutex> guard(lock_);
  return knownWorkerIds_;
};

void DistAutogradContext::addKnownWorkerId(const rpc::worker_id_t workerId) {
  std::lock_guard<std::mutex> guard(lock_);
  knownWorkerIds_.insert(workerId);
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
  std::lock_guard<std::mutex> guard(lock_);
  return sendAutogradFunctions_;
}

std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
DistAutogradContext::recvFunctions() const {
  std::lock_guard<std::mutex> guard(lock_);
  return recvAutogradFunctions_;
}

void DistAutogradContext::accumulateGrad(
    const torch::autograd::Variable& variable,
    const torch::Tensor& grad,
    size_t num_expected_refs) {
  TORCH_INTERNAL_ASSERT(grad.defined());
  TORCH_INTERNAL_ASSERT(variable.requires_grad());

  std::lock_guard<std::mutex> guard(lock_);
  auto it = accumulatedGrads_.find(variable);
  at::Tensor old_grad;
  if (it != accumulatedGrads_.end()) {
    // Accumulate multiple grads on the same variable.
    old_grad = it->value();
  }

  // No higher order gradients supported in distributed autograd.
  AutoGradMode grad_mode(false);
  // TODO: Need to bump 'num_expected_refs' here when we support post_hooks for
  // distributed autograd as part of
  // https://github.com/pytorch/pytorch/issues/33482
  AccumulateGrad::accumulateGradAndCallHooks(
      variable,
      old_grad,
      grad,
      // Add +1 here since we can't std::move(grad) since it is a const ref,
      // which incurs a refcount bump for the Tensor.
      num_expected_refs + 1,
      [this, &variable](at::Tensor&& grad_update) {
        accumulatedGrads_.insert(variable, std::move(grad_update));
      });
}

std::shared_ptr<torch::autograd::GraphTask> DistAutogradContext::
    retrieveGraphTask() {
  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(graphTask_);
  return graphTask_;
}

void DistAutogradContext::setGraphTask(
    std::shared_ptr<torch::autograd::GraphTask> graphTask) {
  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      !graphTask_,
      "Cannot set GraphTask multiple times for the same autograd context");
  graphTask_ = std::move(graphTask);
}

void DistAutogradContext::resetGraphTask() {
  std::lock_guard<std::mutex> guard(lock_);
  graphTask_ = nullptr;
}

void DistAutogradContext::addOutstandingRpc(
    const std::shared_ptr<rpc::FutureMessage>& futureMessage) {
  futureMessage->addCallback(
      [this](
          const rpc::Message& /* unused */,
          const c10::optional<utils::FutureError>& futErr) {
        if (futErr) {
          // If we have an error, let the local autograd engine know about it.
          std::runtime_error err((*futErr).what());
          std::unique_lock<std::mutex> lock(lock_);
          if (graphTask_) {
            graphTask_->set_exception_without_signal(nullptr);
            lock.unlock();
            graphTask_->future_result_->setErrorIfNeeded(err.what());
          } else {
            LOG(WARNING)
                << "Ignoring error since GraphTask is no longer valid: "
                << err.what();
          }
        }
      });
  std::lock_guard<std::mutex> guard(lock_);
  outStandingRpcs_.push_back(futureMessage);
}

void DistAutogradContext::clearOutstandingRpcs() {
  std::unique_lock<std::mutex> lock(lock_);
  outStandingRpcs_.clear();
}

std::shared_ptr<rpc::FutureMessage> DistAutogradContext::
    clearAndWaitForOutstandingRpcsAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  auto outStandingRpcs = std::move(outStandingRpcs_);
  lock.unlock();

  struct State {
    explicit State(int32_t count)
        : future(std::make_shared<rpc::FutureMessage>()), remaining(count) {}
    std::shared_ptr<rpc::FutureMessage> future;
    std::atomic<int32_t> remaining;
    std::atomic<bool> alreadySentError{false};
  };
  auto state = std::make_shared<State>(outStandingRpcs.size());
  if (outStandingRpcs.empty()) {
    state->future->markCompleted(rpc::Message());
  } else {
    for (auto& rpc : outStandingRpcs) {
      rpc->addCallback([state](
                           const rpc::Message& /* unused */,
                           const c10::optional<utils::FutureError>& err) {
        if (err) {
          // If there's an error, we want to setError() on the future, unless
          // another error has already been sent - use a CAS to guard.
          //
          // Don't decrement num remaining here! (We don't need to, since memory
          // handling is separate). If we simply don't decrement on errors,
          // reaching 0 means that there were no errors - and hence, we can just
          // markCompleted() without any other checking there.
          bool expectedAlreadySent = false;
          if (state->alreadySentError.compare_exchange_strong(
                  expectedAlreadySent, true)) {
            state->future->setError(err->what());
          }
          return;
        }

        if (--state->remaining == 0) {
          state->future->markCompleted(rpc::Message());
        }
      });
    }
  }
  return state->future;
}

std::shared_ptr<SendRpcBackward> DistAutogradContext::retrieveSendFunction(
    int64_t autograd_message_id) {
  std::lock_guard<std::mutex> guard(lock_);
  auto it = sendAutogradFunctions_.find(autograd_message_id);
  TORCH_CHECK(
      it != sendAutogradFunctions_.end(),
      "Could not find send function for autograd message id: ",
      autograd_message_id);
  return it->second;
}

const c10::Dict<torch::Tensor, torch::Tensor> DistAutogradContext::
    getGradients() const {
  std::lock_guard<std::mutex> guard(lock_);
  return accumulatedGrads_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
