#include <torch/csrc/distributed/autograd/context/context.h>

#include <functional>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

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

  at::Tensor new_grad = AccumulateGrad::callHooks(variable, grad);

  // TODO: Need to bump 'num_expected_refs' here when we support post_hooks for
  // distributed autograd as part of
  // https://github.com/pytorch/pytorch/issues/33482
  AccumulateGrad::accumulateGrad(
      variable,
      old_grad,
      new_grad,
      // Add +1 here since we can't std::move(grad) when call
      // AccumulateGrad::callHooks, since it is a const ref, and that incurs a
      // refcount bump for the new_grad.
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
    const std::shared_ptr<rpc::JitFuture>& jitFuture) {
  std::weak_ptr<rpc::JitFuture> wp = jitFuture;
  jitFuture->addCallback([this, wp]() {
    auto future = wp.lock();
    if (future->hasError()) {
      // If we have an error, let the local autograd engine know about it.
      std::unique_lock<std::mutex> lock(lock_);
      if (graphTask_) {
        graphTask_->set_exception_without_signal(nullptr);
        lock.unlock();
        if (!graphTask_->future_completed_.exchange(true)) {
          graphTask_->future_result_->setErrorIfNeeded(future->exception_ptr());
        }
      } else {
        LOG(WARNING) << "Ignoring error since GraphTask is no longer valid: "
                     << future->tryRetrieveErrorMessage();
      }
    }
  });
  std::lock_guard<std::mutex> guard(lock_);
  outStandingRpcs_.push_back(jitFuture);
}

void DistAutogradContext::clearOutstandingRpcs() {
  std::unique_lock<std::mutex> lock(lock_);
  outStandingRpcs_.clear();
}

std::shared_ptr<c10::ivalue::Future> DistAutogradContext::
    clearAndWaitForOutstandingRpcsAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  auto outStandingRpcs = std::move(outStandingRpcs_);
  lock.unlock();

  struct State {
    explicit State(int32_t count)
        : future(
              std::make_shared<c10::ivalue::Future>(c10::NoneType::create())),
          remaining(count) {}
    std::shared_ptr<c10::ivalue::Future> future;
    std::atomic<int32_t> remaining;
    std::atomic<bool> alreadySentError{false};
  };
  auto state = std::make_shared<State>(outStandingRpcs.size());
  if (outStandingRpcs.empty()) {
    state->future->markCompleted(c10::IValue());
  } else {
    for (auto& rpc : outStandingRpcs) {
      std::weak_ptr<rpc::JitFuture> wp = rpc;
      rpc->addCallback([state, wp]() {
        auto future = wp.lock();
        if (future->hasError()) {
          // If there's an error, we want to setError() on the future,
          // unless another error has already been sent - use a CAS to
          // guard.
          //
          // Don't decrement num remaining here! (We don't need to, since
          // memory handling is separate). If we simply don't decrement on
          // errors, reaching 0 means that there were no errors - and hence,
          // we can just markCompleted() without any other checking there.
          bool expectedAlreadySent = false;
          if (state->alreadySentError.compare_exchange_strong(
                  expectedAlreadySent, true)) {
            state->future->setError(future->exception_ptr());
          }
          return;
        }

        if (--state->remaining == 0) {
          state->future->markCompleted(c10::IValue());
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

void DistAutogradContext::runGradCallbackForVariable(
    const torch::autograd::Variable& variable,
    GradCallback&& cb) {
  torch::Tensor grad;
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto it = accumulatedGrads_.find(variable);
    TORCH_INTERNAL_ASSERT(
        it != accumulatedGrads_.end(),
        "The grad for the variable should exist in dist_autograd context.");
    grad = it->value();
  }
  if (cb(grad)) {
    std::lock_guard<std::mutex> guard(lock_);
    // Needs to update the grad in the map.
    accumulatedGrads_.insert_or_assign(variable, std::move(grad));
  }
}

namespace {
thread_local ContextPtr tl_context_ptr;
} // namespace

ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext(
    ContextPtr&& new_context)
    : prev_context_ptr_(std::move(tl_context_ptr)) {
  tl_context_ptr = std::move(new_context);
}

ThreadLocalDistAutogradContext::~ThreadLocalDistAutogradContext() {
  tl_context_ptr = std::move(prev_context_ptr_);
}

// static
ContextPtr ThreadLocalDistAutogradContext::getContextPtr() {
  return tl_context_ptr;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
