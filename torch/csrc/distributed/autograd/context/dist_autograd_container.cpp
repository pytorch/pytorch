#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr int kAutoIncrementBits = 48;
constexpr int64_t kAutoIncrementMask = (1LL << kAutoIncrementBits) - 1;
constexpr int kMaxWorkerId = 65535;
/* const std::chrono::duration<double> kContextTimeout = std::chrono::seconds(2); */
const auto kContextTimeout = std::chrono::seconds(2);
const auto kcleanupWatchdogCVTimeout = std::chrono::seconds(60);

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
      max_id_(0) {
  cleanupContextTimeout = std::chrono::seconds(600);
  terminateWatchdog_.store(false);
  cleanupWatchdogThread_ = std::thread(&DistAutogradContainer::cleanupContextWatchdog, this);
  LOG(ERROR) << getWorkerId() << "- watchdog thread created\n";
  /* creation_time = std::chrono::high_resolution_clock::now(); */
}

DistAutogradContainer::~DistAutogradContainer() {
  LOG(ERROR) << getWorkerId() << "- destructor called\n";
  terminateWatchdog_.store(true);
  cleanupWatchdogCV_.notify_one();
  cleanupWatchdogThread_.join();
  LOG(ERROR) << getWorkerId() << "- thread joined\n";
}
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

void DistAutogradContainer::setCleanupContextTimeout(int64_t newTimeout) {
  TORCH_CHECK(
      newTimeout >=0,
      "cleanupContextTimeout must be nonnegative");
  cleanupContextTimeout = std::chrono::seconds(newTimeout);
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
    return std::get<0>(it->second);
  }

  auto& pair = autograd_context_
                      .emplace(
                          std::piecewise_construct,
                          std::forward_as_tuple(context_id),
                          std::forward_as_tuple(context_id, std::chrono::high_resolution_clock::now()))
                      .first->second;
  auto& context = std::get<0>(pair);
  auto now = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - std::get<1>(autograd_context_.at(context_id)));
  LOG(ERROR) << getWorkerId() << "- insertion, new size: " << autograd_context_.size() << " context_id: " << context_id << " time: " << diff.count() << " getcreate\n";
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

  auto& pair = autograd_context_
                      .emplace(
                          std::piecewise_construct,
                          std::forward_as_tuple(next_context_id_),
                          std::forward_as_tuple(next_context_id_, std::chrono::high_resolution_clock::now()))
                      .first->second;
  auto& context = std::get<0>(pair);
  auto now = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - std::get<1>(autograd_context_.at(next_context_id_)));
  LOG(ERROR) << getWorkerId() << "- insertion, new size: " << autograd_context_.size() << " context_id: " << next_context_id_ << " time: " << diff.count() << " insert\n";
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
      "data for current autograd context id."
      "Worker Id: ", getWorkerId());
  return std::get<0>(it->second);
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
      std::get<0>(autograd_context_.find(context_id)->second).getKnownWorkerIds();
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

void DistAutogradContainer::cleanupContextWatchdog() {
  /* LOG(ERROR) << getWorkerId() << "- scheduled again! watchdog: " << terminateWatchdog_.load() <<  "\n"; */
  /* std::this_thread::sleep_for(cleanupContextTimeout); */
  while (!terminateWatchdog_.load()) {
    {
      std::lock_guard<std::mutex> guard(autograd_context_lock_);
      /* LOG(ERROR) << getWorkerId() << "- got inside\n"; */
      /* if (autograd_context_.empty()) { */
      /*   LOG(ERROR) << getWorkerId() << "- nothing to loop thru\n"; */
      /* } */
      for (auto it = autograd_context_.begin(); it != autograd_context_.end();) {
        auto now = std::chrono::high_resolution_clock::now();
        /* LOG(ERROR) << getWorkerId() << "- candidate id " << it->first << "\n"; */
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - std::get<1>(it->second));
        /* LOG(ERROR) << getWorkerId() << "- curr time: " << diff.count() << "\n"; */
        if (diff >= cleanupContextTimeout) { // && autograd_context_.find(pair.first) != autograd_context_.end()) {
          // TODO: see if you can just invoke destructor
          /* LOG(ERROR) << getWorkerId() << "- cleared timeout!\n"; */
          LOG(ERROR) << getWorkerId() << "- DELETING " << it->first << "\n";
          /* eraseContextIdAndReset(pair.first); */
          auto context_id = it->first;
          it = autograd_context_.erase(it);
          if (current_context_id_ == context_id) {
            // Reset the thread_local current context id, since it is no longer valid.
            current_context_id_ = kInvalidContextId;
          }
          LOG(ERROR) << getWorkerId() << "- queue size post-deletion is " << autograd_context_.size() << "\n";
        }
      }
    }
    // CV stuff
    /* LOG(ERROR) << getWorkerId() << "- Waiting for destruction\n"; */
    /* std::unique_lock<std::mutex> guard(cleanupWatchdogCVMutex_); */
    /* cleanupWatchdogCV_.wait_for( */
    /*     guard, */
    /*     std::chrono::seconds(300), */
    /*     [&]() -> bool { return terminateWatchdog_.load(); }); */
  }
  LOG(ERROR) << getWorkerId() << "- Waiting for destruction\n";
  std::unique_lock<std::mutex> guard(cleanupWatchdogCVMutex_);
  cleanupWatchdogCV_.wait_for(
      guard,
      kcleanupWatchdogCVTimeout,
      [&]() -> bool { return terminateWatchdog_.load(); });
}

/* void DistAutogradContainer::cleanupContextWatchdog() { */
/*   LOG(ERROR) << getWorkerId() << "- scheduled again!\n"; */
/*   std::this_thread::sleep_for(cleanupContextTimeout); */
/*   while(!terminateWatchdog_.load()) { */
/*     std::lock_guard<std::mutex> guard(autograd_context_lock_); */
/*     for (auto& pair : autograd_context_) { */
/*       /1* if (autograd_context_.find(pair.first) != autograd_context_.end()) { *1/ */ 
/*         LOG(ERROR) << "just in case " << pair.first << "\n"; */
/*         sendReleaseContextRpc(pair.first); */
/*         eraseContextIdAndReset(pair.first); */
/*         LOG(ERROR) << getWorkerId() << "- queue size post-deletion is " << autograd_context_.size() << "\n"; */
/*         if (autograd_context_.empty()) { */
/*           return; */
/*         } */
/*       /1* } *1/ */
/*     } */
/*   } */
/* } */

DistAutogradContext& DistAutogradContainer::retrieveContext(
    int64_t context_id) {
  std::lock_guard<std::mutex> guard(autograd_context_lock_);
  TORCH_CHECK(
      autograd_context_.find(context_id) != autograd_context_.end(),
      "Could not find autograd context with id: ",
      context_id);
  return std::get<0>(autograd_context_.at(context_id));
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
