#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <c10/util/Exception.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr int kContextIdBits = 48;
constexpr int64_t kContextIdMask = (1LL << kContextIdBits) - 1;
constexpr int kMaxWorkerId = 65535;
constexpr int64_t kMaxContextId = kContextIdMask;

thread_local int64_t DistAutogradContainer::current_context_id_ = -1;

DistAutogradContainer::DistAutogradContainer()
    : next_context_id_(0), worker_id_(0), initialized_(false) {}

DistAutogradContainer& DistAutogradContainer::init(int64_t worker_id) {
  TORCH_CHECK(
      worker_id >= 0 && worker_id <= kMaxWorkerId,
      "worker_id needs to be in the range [0, 65535]")

  auto& container = getInstance();
  container.worker_id_ = worker_id;
  container.next_context_id_ = static_cast<int64_t>(worker_id)
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
  TORCH_INTERNAL_ASSERT(
      next_context_id_ < std::numeric_limits<int64_t>::max() &&
          next_context_id_ <
              (kMaxContextId |
               (static_cast<int64_t>(worker_id_) << kContextIdBits)),
      "We have run out of autograd context ids!!!");

  autograd_context_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(next_context_id_),
      std::forward_as_tuple(next_context_id_));

  current_context_id_ = next_context_id_;
  return autograd_context_.at(next_context_id_++);
}

bool DistAutogradContainer::hasValidContext() const {
  return current_context_id_ != -1;
}

DistAutogradContext& DistAutogradContainer::currentContext() {
  TORCH_CHECK(
      hasValidContext(),
      "Current thread doesn't have a valid autograd context. Please wrap your "
      "code using: `with torch.distributed.autograd.context() as context_id` "
      "to generate a valid context");
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  auto it = autograd_context_.find(current_context_id_);
  TORCH_CHECK(
      it != autograd_context_.end(),
      "Couldn't find autograd context "
      "data for current autograd context id");
  return it->second;
}

void DistAutogradContainer::releaseContext(int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);
  autograd_context_.erase(context_id);

  if (current_context_id_ == context_id) {
    // Reset the thread_local current context id, since it is no longer valid.
    current_context_id_ = -1;
  }
}

const DistAutogradContext& DistAutogradContainer::retrieveContext(
    int64_t context_id) const {
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
