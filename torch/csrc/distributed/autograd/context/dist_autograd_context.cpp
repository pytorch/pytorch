#include <functional>

#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <c10/util/Exception.h>

namespace torch {
namespace distributed {
namespace autograd {

namespace {

// Autograd function used to enqueue an error on the local autograd engine.
class ErrorFunc : public torch::autograd::Node {
 public:
  explicit ErrorFunc(const std::exception& error) : error_(error) {}

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    throw error_;
  }

 private:
  std::exception error_;
};

} // anonymous namespace.

DistAutogradContext::DistAutogradContext(int64_t contextId)
    : contextId_(contextId) {}

int64_t DistAutogradContext::contextId() const {
  return contextId_;
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
    const torch::Tensor& grad) {
  TORCH_INTERNAL_ASSERT(grad.defined());
  TORCH_INTERNAL_ASSERT(variable.requires_grad());

  std::lock_guard<std::mutex> guard(lock_);
  auto it = accumulatedGrads_.find(variable);
  if (it != accumulatedGrads_.end()) {
    // Accumulate multiple grads on the same variable.
    it->value().add_(grad);
  } else {
    // First grad for this variable.
    accumulatedGrads_.insert(variable, grad);
  }
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

void DistAutogradContext::addOutstandingRpc(
    const std::shared_ptr<rpc::FutureMessage>& futureMessage) {
  futureMessage->addCallback(std::bind(
      &DistAutogradContext::outStandingRpcCallback,
      this,
      std::placeholders::_1));
  std::lock_guard<std::mutex> guard(lock_);
  outStandingRpcs_.push_back(futureMessage);
}

void DistAutogradContext::outStandingRpcCallback(const rpc::Message& message) {
  if (message.type() == rpc::MessageType::EXCEPTION) {
    // If we have an error, let the local autograd engine know about it.
    std::string err(message.payload().begin(), message.payload().end());
    auto exception = std::runtime_error(err);

    // Enqueue 'ErrorFunc' on the local autograd engine.
    auto& localEngine = torch::autograd::Engine::get_default_engine();
    auto errorFunc = std::make_shared<ErrorFunc>(exception);

    // Increment out standing tasks for this function and set appropriate
    // exec_info_ for this function to execute.
    graphTask_->outstanding_tasks_++;
    // Lock mutex for writing to exec_info_.
    std::lock_guard<std::mutex> lock(graphTask_->mutex_);
    graphTask_->exec_info_[errorFunc.get()].needed_ = true;
    localEngine.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask_.get(), errorFunc, torch::autograd::InputBuffer(0)));
  }
}

void DistAutogradContext::clearAndWaitForOutstandingRpcs() {
  // Copy futures under lock, but wait for them outside the lock.
  std::unique_lock<std::mutex> lock(lock_);
  auto outStandingRpcs = std::move(outStandingRpcs_);
  lock.unlock();

  for (const auto& outStandingRpc : outStandingRpcs) {
    outStandingRpc->wait();
  }
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
