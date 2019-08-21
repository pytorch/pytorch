#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <c10/util/Exception.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr int kContextIdBits = 48;
constexpr int64_t kContextIdMask = (1LL << kContextIdBits) - 1;
constexpr int kMaxWorkerId = 65535;
constexpr int64_t kMaxContextId = kContextIdMask;

DistAutogradContainer::DistAutogradContainer() : initialized_(false) {}

DistAutogradContainer& DistAutogradContainer::init(int64_t worker_id) {
  TORCH_CHECK(
      worker_id >= 0 && worker_id <= kMaxWorkerId,
      "worker_id needs to be in the range [0, 65535]")

  auto& container = getInstance();
  container.worker_id_ = worker_id;
  container.current_context_id_ = static_cast<int64_t>(worker_id)
      << kContextIdBits;
  container.initialized_ = true;
  return container;
}

DistAutogradContainer& DistAutogradContainer::getInstance() {
  static DistAutogradContainer container;
  return container;
}

const DistAutogradContext& DistAutogradContainer::newContext() {
  if (!initialized_) {
    throw std::runtime_error(
        "Need to initialize distributed autograd using "
        "torch.distributed.autograd.init()");
  }

  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  if (current_context_id_ == std::numeric_limits<int64_t>::max() ||
      current_context_id_ >
          (kMaxContextId |
           (static_cast<int64_t>(worker_id_) << kContextIdBits))) {
    throw std::runtime_error("We have run out of autograd context ids!!!");
  }

  autograd_context_.emplace(
      current_context_id_, DistAutogradContext(current_context_id_));
  return autograd_context_.at(current_context_id_++);
}

void DistAutogradContainer::releaseContext(int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);
  autograd_context_.erase(context_id);
}

const DistAutogradContext& DistAutogradContainer::retrieveContext(
    int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);
  return autograd_context_.at(context_id);
}

} // namespace autograd
} // namespace distributed
} // namespace torch
