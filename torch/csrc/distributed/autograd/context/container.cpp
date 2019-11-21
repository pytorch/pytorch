#include <torch/csrc/distributed/autograd/context/container.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr int kAutoIncrementBits = 48;
constexpr int64_t kAutoIncrementMask = (1LL << kAutoIncrementBits) - 1;
constexpr int kMaxWorkerId = 65535;

constexpr int64_t kInvalidContextId = -1;

// Each thread has a single autograd_context_id valid at any point in time.
static thread_local int64_t current_context_id_ = kInvalidContextId;

// Lock to ensure DistAutogradContainer is initialized only once.
static std::mutex dist_container_init_lock_;

DistAutogradContainer::DistAutogradContainer()
    : next_context_id_(0),
      worker_id_(0),
      initialized_(false),
      next_autograd_message_id_(0),
      max_id_(0) {}

DistAutogradContainer& DistAutogradContainer::init(int64_t worker_id) {
  std::lock_guard<std::mutex> guard(dist_container_init_lock_);

  TORCH_CHECK(
      worker_id >= 0 && worker_id <= kMaxWorkerId,
      "worker_id needs to be in the range [0, 65535]")

  auto& container = getInstanceInternal();
  TORCH_CHECK(
      !container.initialized_,
      "Container is already initialized! Cannot initialize it twice!");

  container.worker_id_ = worker_id;
  container.next_context_id_ = static_cast<int64_t>(worker_id)
      << kAutoIncrementBits;
  container.next_autograd_message_id_ = static_cast<int64_t>(worker_id)
      << kAutoIncrementBits;
  container.max_id_ =
      (kAutoIncrementMask |
       (static_cast<int64_t>(worker_id) << kAutoIncrementBits));
  container.initialized_ = true;
  return container;
}

DistAutogradContainer& DistAutogradContainer::getInstance() {
  auto& instance = getInstanceInternal();
  TORCH_CHECK(
      instance.initialized_,
      "Need to initialize distributed autograd using "
      "torch.distributed.autograd.init()");
  return instance;
}

DistAutogradContainer& DistAutogradContainer::getInstanceInternal() {
  static DistAutogradContainer container;
  return container;
}

int64_t DistAutogradContainer::newAutogradMessageId() {
  // Check for overflow into workerId_ section.
  TORCH_INTERNAL_ASSERT(next_autograd_message_id_ < max_id_);
  return next_autograd_message_id_++;
}

DistAutogradContext& DistAutogradContainer::getOrCreateContext(
    int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  auto it = autograd_context_.find(context_id);
  if (it != autograd_context_.end()) {
    return it->second;
  }

  auto& context = autograd_context_
                      .emplace(
                          std::piecewise_construct,
                          std::forward_as_tuple(context_id),
                          std::forward_as_tuple(context_id))
                      .first->second;
  return context;
}

rpc::worker_id_t DistAutogradContainer::getWorkerId() const {
  return worker_id_;
}

const DistAutogradContext& DistAutogradContainer::newContext() {
  TORCH_CHECK(
      current_context_id_ == kInvalidContextId,
      "Already have an autograd context id for this thread.");

  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  // Check for overflow into workerId_ section.
  TORCH_INTERNAL_ASSERT(next_context_id_ < max_id_);

  auto& context = autograd_context_
                      .emplace(
                          std::piecewise_construct,
                          std::forward_as_tuple(next_context_id_),
                          std::forward_as_tuple(next_context_id_))
                      .first->second;

  current_context_id_ = next_context_id_++;
  return context;
}

bool DistAutogradContainer::hasValidContext() const {
  return current_context_id_ != kInvalidContextId;
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

void DistAutogradContainer::releaseContextIfPresent(int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  // no-op if the context does not exist on this thread. This could happen if an
  // in-flight RPC has already released the context on this thread.
  if (autograd_context_.find(context_id) == autograd_context_.end()) {
    return;
  }
  sendReleaseContextRpc(context_id);
  eraseContextIdAndReset(context_id);
}

void DistAutogradContainer::releaseContext(int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);

  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);

  sendReleaseContextRpc(context_id);
  eraseContextIdAndReset(context_id);
}

void DistAutogradContainer::sendReleaseContextRpc(int64_t context_id) {
  // notify other workers to clean up their contexts.
  auto workerIds =
      autograd_context_.find(context_id)->second.getKnownWorkerIds();
  auto agent = rpc::RpcAgent::getDefaultRpcAgent();
  for (const auto& worker_id : workerIds) {
    agent->send(
        agent->getWorkerInfo(worker_id),
        CleanupAutogradContextReq(context_id).toMessage());
  }
}

void DistAutogradContainer::eraseContextIdAndReset(int64_t context_id) {
  autograd_context_.erase(context_id);

  if (current_context_id_ == context_id) {
    // Reset the thread_local current context id, since it is no longer valid.
    current_context_id_ = kInvalidContextId;
  }
}

DistAutogradContext& DistAutogradContainer::retrieveContext(
    int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);
  return autograd_context_.at(context_id);
}

int64_t DistAutogradContainer::getMaxId() {
  return max_id_;
}

void DistAutogradContainer::setCurrentContextId(int64_t contextId) {
  TORCH_INTERNAL_ASSERT(
      current_context_id_ == kInvalidContextId,
      "Already have an autograd context id for this thread.");
  current_context_id_ = contextId;
}

void DistAutogradContainer::clearCurrentContext() {
  current_context_id_ = -1;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
