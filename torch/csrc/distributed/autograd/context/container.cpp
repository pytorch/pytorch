#include <torch/csrc/distributed/autograd/context/container.h>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>

namespace torch::distributed::autograd {

constexpr int kAutoIncrementBits = 48;
constexpr int64_t kAutoIncrementMask = (1LL << kAutoIncrementBits) - 1;
constexpr int kMaxWorkerId = 65535;
constexpr int kNumCleanupContextRetries = 20;

constexpr int64_t kInvalidContextId = -1;

// Each thread has a single autograd_context_id valid at any point in time.
static thread_local int64_t current_context_id_ = kInvalidContextId;

// Lock to ensure DistAutogradContainer is initialized only once.
static std::mutex dist_container_init_lock_;

DistAutogradContainer::DistAutogradContainer(uint32_t num_shards)
    : next_context_id_(0),
      worker_id_(0),
      initialized_(false),
      autograd_contexts_(num_shards),
      num_shards_(num_shards),
      next_autograd_message_id_(0),
      max_id_(0) {
  // num_shards has to be a power of 2 for the modulo trick in 'getShard'
  // to work.
  TORCH_INTERNAL_ASSERT((num_shards & (num_shards - 1)) == 0);
}

DistAutogradContainer& DistAutogradContainer::init(int64_t worker_id) {
  std::lock_guard<std::mutex> guard(dist_container_init_lock_);

  TORCH_CHECK(
      worker_id >= 0 && worker_id <= kMaxWorkerId,
      "worker_id needs to be in the range [0, 65535]")

  auto& container = getInstanceInternal();
  TORCH_CHECK(
      !container.initialized_ || (worker_id == container.worker_id_),
      "Container is already initialized with worker_id: ",
      container.worker_id_,
      ", cannot initialize with different worker_id: ",
      worker_id);

  if (container.initialized_) {
    LOG(INFO) << "DistAutogradContainer is already initialized";
    return container;
  }

  container.worker_id_ = static_cast<int16_t>(worker_id);
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

uint32_t DistAutogradContainer::computeNumShards() {
  uint32_t num_shards = 1;
  auto num_hw_threads = std::thread::hardware_concurrency();
  if (num_hw_threads == 0) {
    num_shards = kNumDefaultShards;
  } else {
    // Compute the next power of 2 which is higher than twice the hardware
    // concurrency.
    while (num_shards < num_hw_threads * 2) {
      num_shards <<= 1;
    }
  }
  VLOG(1) << "Number of shards for DistAutogradContainer: " << num_shards;
  return num_shards;
}

inline DistAutogradContainer::ContextsShard& DistAutogradContainer::getShard(
    int64_t context_id) {
  // num_shards_ has to be a power of 2 for this modulo trick to work (validated
  // during init).
  return autograd_contexts_[context_id & (num_shards_ - 1)];
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
  // Leaky singleton to avoid module destructor race.
  static DistAutogradContainer* container =
      new DistAutogradContainer(computeNumShards());
  return *container;
}

int64_t DistAutogradContainer::newAutogradMessageId() {
  // Check for overflow into workerId_ section.
  TORCH_INTERNAL_ASSERT(next_autograd_message_id_ < max_id_);
  return next_autograd_message_id_++;
}

ContextPtr DistAutogradContainer::getOrCreateContext(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::lock_guard<std::mutex> guard(shard.lock);
  auto it = shard.contexts.find(context_id);
  if (it != shard.contexts.end()) {
    return it->second;
  }

  auto& context =
      shard.contexts
          .emplace(
              std::piecewise_construct,
              std::forward_as_tuple(context_id),
              std::forward_as_tuple(
                  std::make_shared<DistAutogradContext>(context_id)))
          .first->second;
  return context;
}

rpc::worker_id_t DistAutogradContainer::getWorkerId() const {
  return worker_id_;
}

const ContextPtr DistAutogradContainer::newContext() {
  TORCH_CHECK(
      current_context_id_ == kInvalidContextId,
      "Already have an autograd context id for this thread.");

  auto context_id = next_context_id_++;
  current_context_id_ = context_id;

  // Check for overflow into workerId_ section.
  TORCH_INTERNAL_ASSERT(context_id < max_id_);

  auto& shard = getShard(context_id);
  std::lock_guard<std::mutex> guard(shard.lock);
  auto& context =
      shard.contexts
          .emplace(
              std::piecewise_construct,
              std::forward_as_tuple(context_id),
              std::forward_as_tuple(
                  std::make_shared<DistAutogradContext>(context_id)))
          .first->second;

  return context;
}

bool DistAutogradContainer::hasValidContext() const {
  return current_context_id_ != kInvalidContextId;
}

ContextPtr DistAutogradContainer::currentContext() {
  TORCH_CHECK(
      hasValidContext(),
      "Current thread doesn't have a valid autograd context. Please wrap your "
      "code using: `with torch.distributed.autograd.context() as context_id` "
      "to generate a valid context");

  auto& shard = getShard(current_context_id_);
  std::lock_guard<std::mutex> guard(shard.lock);
  auto it = shard.contexts.find(current_context_id_);
  TORCH_CHECK(
      it != shard.contexts.end(),
      "Couldn't find autograd context "
      "data for current autograd context id");
  return it->second;
}

void DistAutogradContainer::releaseContextIfPresent(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::unique_lock<std::mutex> lock(shard.lock);
  auto it = shard.contexts.find(context_id);

  // no-op if the context does not exist on this thread. This could happen if an
  // in-flight RPC has already released the context on this thread.
  if (it == shard.contexts.end()) {
    return;
  }

  auto knownWorkerIds = it->second->getKnownWorkerIds();
  eraseContextIdAndReset(shard, context_id);

  // Unlock since we no longer need the lock.
  lock.unlock();
  sendReleaseContextRpc(knownWorkerIds, context_id);
}

void DistAutogradContainer::releaseContext(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::unique_lock<std::mutex> lock(shard.lock);
  auto it = shard.contexts.find(context_id);

  TORCH_CHECK(
      it != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);

  auto knownWorkerIds = it->second->getKnownWorkerIds();
  eraseContextIdAndReset(shard, context_id);

  // Unlock since we no longer need the lock.
  lock.unlock();
  sendReleaseContextRpc(knownWorkerIds, context_id);
}

void DistAutogradContainer::sendReleaseContextRpc(
    const std::unordered_set<rpc::worker_id_t>& workerIds,
    int64_t context_id) {
  // Best-effort notification to other workers to clean up their Dist autograd
  // context, in order to reduce memory usage.
  // agent.send() or getCurrentRpcAgent may throw an error in the case of an
  // ungraceful shutdown, where we are shutting down RPC and also processing
  // this message in a separate thread concurrently. In this case, don't throw
  // here.
  std::shared_ptr<rpc::RpcAgent> agent;
  try {
    agent = rpc::RpcAgent::getCurrentRpcAgent();
  } catch (const std::exception& e) {
    LOG(INFO)
        << "Failed to send RPC to clear Dist Autograd context to all workers: "
        << e.what();
    return;
  }

  TORCH_INTERNAL_ASSERT(agent, "RPC Agent should be set.");

  rpc::RpcRetryOptions options;
  options.maxRetries = kNumCleanupContextRetries;
  for (const auto& worker_id : workerIds) {
    try {
      auto cleanupFuture = agent->sendWithRetries(
          agent->getWorkerInfo(worker_id),
          CleanupAutogradContextReq(context_id).toMessage(),
          options);

      cleanupFuture->addCallback([worker_id](rpc::JitFuture& future) {
        if (future.hasError()) {
          std::string errorMsg = c10::str(
              "Could not release Dist Autograd Context on node ",
              worker_id,
              ": ",
              future.tryRetrieveErrorMessage());
          LOG(ERROR) << errorMsg;
          return;
        }
      });
    } catch (const std::exception& e) {
      LOG(INFO)
          << "Failed to send RPC to clear Dist Autograd context to worker id: "
          << worker_id << " : " << e.what();
    }
  }
}

void DistAutogradContainer::eraseContextIdAndReset(
    DistAutogradContainer::ContextsShard& shard,
    int64_t context_id) {
  // We already have the shard lock here.
  shard.contexts.erase(context_id);

  if (current_context_id_ == context_id) {
    // Reset the thread_local current context id, since it is no longer valid.
    current_context_id_ = kInvalidContextId;
  }
}

void DistAutogradContainer::isValidContext(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::lock_guard<std::mutex> guard(shard.lock);
  TORCH_CHECK(
      shard.contexts.find(context_id) != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);
}

ContextPtr DistAutogradContainer::retrieveContext(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::lock_guard<std::mutex> guard(shard.lock);
  auto it = shard.contexts.find(context_id);
  TORCH_CHECK(
      it != shard.contexts.end(),
      "Could not find autograd context with id: ",
      context_id);
  return it->second;
}

int64_t DistAutogradContainer::getMaxId() {
  return max_id_;
}

void DistAutogradContainer::forceCurrentContextId(int64_t contextId) {
  current_context_id_ = contextId;
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

size_t DistAutogradContainer::numAutogradContexts() const {
  size_t ret = 0;
  for (const auto& shard : autograd_contexts_) {
    std::lock_guard<std::mutex> guard(shard.lock);
    ret += shard.contexts.size();
  }
  return ret;
}

int64_t DistAutogradContainer::currentContextId() {
  return current_context_id_;
}

} // namespace torch::distributed::autograd
