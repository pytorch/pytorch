#include <c10/util/DeadlockDetection.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerInfo::MAX_NAME_LEN;

// Large Time Duration for waiting on the condition variable until the map is
// population. Cannot use
// std::chrono::time_point<std::chrono::steady_clock>::max() due to a known
// overflow-related bug.
constexpr auto kLargeTimeDuration = std::chrono::hours(10000);

RpcAgent::RpcAgent(
    WorkerInfo workerId,
    std::unique_ptr<RequestCallback> cb,
    std::chrono::milliseconds rpcTimeout)
    : workerInfo_(std::move(workerId)),
      cb_(std::move(cb)),
      rpcTimeout_(rpcTimeout),
      profilingEnabled_(false),
      rpcAgentRunning_(false) {}

RpcAgent::~RpcAgent() {
  if (rpcAgentRunning_.load()) {
    shutdown();
  }
}

void RpcAgent::start() {
  rpcAgentRunning_.store(true);
  rpcRetryThread_ = std::thread(&RpcAgent::retryExpiredRpcs, this);
  startImpl();
}

void RpcAgent::shutdown() {
  TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP();
  std::unique_lock<std::mutex> lock(rpcRetryMutex_);
  rpcAgentRunning_.store(false);
  lock.unlock();
  rpcRetryMapCV_.notify_one();
  if (rpcRetryThread_.joinable()) {
    rpcRetryThread_.join();
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.PureVirtualCall)
  shutdownImpl();
}

c10::intrusive_ptr<JitFuture> RpcAgent::sendWithRetries(
    const WorkerInfo& to,
    Message&& message,
    RpcRetryOptions retryOptions) {
  TORCH_CHECK(retryOptions.maxRetries >= 0, "maxRetries cannot be negative.");
  TORCH_CHECK(
      retryOptions.retryBackoff >= 1,
      "maxRetries cannot be exponentially decaying.");
  TORCH_CHECK(
      retryOptions.rpcRetryDuration.count() >= 0,
      "rpcRetryDuration cannot be negative.");

  auto originalFuture =
      c10::make_intrusive<JitFuture>(at::AnyClassType::get(), getDevices());
  steady_clock_time_point newTime =
      computeNewRpcRetryTime(retryOptions, /* retryCount */ 0);
  // Making a copy of the message so it can be retried after the first send.
  Message msgCopy = message;
  auto jitFuture = send(to, std::move(message));
  auto firstRetryRpc = std::make_shared<RpcRetryInfo>(
      to,
      std::move(msgCopy),
      originalFuture,
      /* retryCount */ 0,
      retryOptions);
  jitFuture->addCallback([this, newTime, firstRetryRpc](JitFuture& future) {
    rpcRetryCallback(future, newTime, firstRetryRpc);
  });

  return originalFuture;
}

void RpcAgent::retryExpiredRpcs() {
  // Stores the retried futures so callbacks can be added outside the lock.
  std::vector<
      std::pair<c10::intrusive_ptr<JitFuture>, std::shared_ptr<RpcRetryInfo>>>
      futures;
  // Stores futures and exception messages for non-retriable error-ed futures.
  std::vector<std::pair<c10::intrusive_ptr<JitFuture>, std::string>>
      errorFutures;

  while (rpcAgentRunning_.load()) {
    std::unique_lock<std::mutex> lock(rpcRetryMutex_);

    // We must continue sleeping as long as the RPC Agent is running and when
    // either the Retry Map is empty, or when the Retry Map's earliest expiring
    // RPC is set to be retried in the future.
    steady_clock_time_point earliestTimeout =
        std::chrono::steady_clock::now() + kLargeTimeDuration;

    for (;;) {
      if (!rpcAgentRunning_.load())
        return;
      if (std::chrono::steady_clock::now() >= earliestTimeout)
        break;
      if (!rpcRetryMap_.empty()) {
        earliestTimeout = rpcRetryMap_.begin()->first;
      }
      rpcRetryMapCV_.wait_until(lock, earliestTimeout);
    }

    // Updating these since something may have been added to the map while this
    // thread was sleeping.
    earliestTimeout = rpcRetryMap_.begin()->first;
    auto& earliestRpcList = rpcRetryMap_.begin()->second;

    // We iterate through all the RPC's set to be retried at the current
    // timepoint, resend those RPC's, and add the RPC's and their futures to
    // a list to later attach callbacks. These callbacks either schedule
    // the RPC for a future retry or marks it with success/error depending on
    // the outcome of the current send. Then, we clean up the rpcRetryMap_.
    for (auto it = earliestRpcList.begin(); it != earliestRpcList.end();
         /* no increment */) {
      auto& earliestRpc = *it;
      // Making a copy of the message so it can be retried in the future.
      Message msgCopy = earliestRpc->message_;
      c10::intrusive_ptr<JitFuture> jitFuture;

      // send() will throw an exception if an RPC is retried while the agent is
      // shutdown. We must catch this exception and mark the original future
      // with an error, since this RPC never succeeded and can no longer be
      // retried.
      try {
        jitFuture = send(earliestRpc->to_, std::move(msgCopy));
        futures.emplace_back(jitFuture, earliestRpc);
      } catch (std::exception& e) {
        // We must store the futures and exception messages here and only mark
        // the futures with an error after releasing the lock.
        errorFutures.emplace_back(earliestRpc->originalFuture_, e.what());
      }

      // A callback will be attached to all futures for the retries in this
      // list. Thus they will either be rescheduled for future retries or they
      // will be marked as complete. We can safely delete them from the retry
      // Map for the current timepoint.
      it = earliestRpcList.erase(it);
    }

    // If there are no more RPC's set to be retried at the current timepoint,
    // we can remove the corresponsing unordered_set from the retry map.
    if (earliestRpcList.empty()) {
      rpcRetryMap_.erase(earliestTimeout);
    }

    lock.unlock();
    // We attach callbacks to the futures outside of the lock to prevent
    // potential deadlocks.
    for (const auto& it : futures) {
      auto jitFuture = it.first;
      auto earliestRpc = it.second;
      steady_clock_time_point newTime = computeNewRpcRetryTime(
          earliestRpc->options_, earliestRpc->retryCount_);
      earliestRpc->retryCount_++;

      jitFuture->addCallback([this, newTime, earliestRpc](JitFuture& future) {
        rpcRetryCallback(future, newTime, earliestRpc);
      });
    }
    futures.clear();

    // For exceptions caught while retrying RPC's above, we set those futures
    // with errors now that we have released the lock.
    for (const auto& it : errorFutures) {
      auto errorFuture = it.first;
      auto errorMsg = it.second;
      errorFuture->setError(
          std::make_exception_ptr(std::runtime_error(errorMsg)));
    }
    errorFutures.clear();
  }
}

void RpcAgent::rpcRetryCallback(
    JitFuture& jitFuture,
    steady_clock_time_point newTime,
    std::shared_ptr<RpcRetryInfo> earliestRpc) {
  if (jitFuture.hasError()) {
    // Adding one since we want to include the original send as well and not
    // just the retry count.
    LOG(INFO) << "Send try " << (earliestRpc->retryCount_ + 1) << " failed";
    if (!rpcAgentRunning_.load()) {
      // If the RPC Agent has shutdown, we cannot retry messages. Thus we mark
      // the future with an error since the RPC was never completed
      // successfully.
      std::string errorMessage = c10::str(
          "RPC Agent is no longer running on Node ",
          RpcAgent::getWorkerInfo().id_,
          ". Cannot retry message.");
      earliestRpc->originalFuture_->setError(jitFuture.exception_ptr());
    } else if (earliestRpc->retryCount_ < earliestRpc->options_.maxRetries) {
      // If the previous future completed with an error and we haven't
      // completed maxRetries send attempts, we move the earliestRpc
      // struct to a new time point in the retry map (effectively
      // scheduling it for a future retry.)
      {
        std::lock_guard<std::mutex> retryMapLock(rpcRetryMutex_);
        rpcRetryMap_[newTime].emplace(std::move(earliestRpc));
      }
      // The retry thread waits for the map to be populated. Thus we notify
      // once an item has been added.
      rpcRetryMapCV_.notify_one();
    } else {
      // We have completed maxRetries send attempts. We're now marking
      // the future with an error.
      std::string errorMessage = c10::str(
          "The RPC has not succeeded after the specified number of max retries (",
          earliestRpc->options_.maxRetries,
          ").");
      earliestRpc->originalFuture_->setError(
          std::make_exception_ptr(std::runtime_error(errorMessage)));
    }
  } else {
    // This try succeeded, so we can make the original future as complete.
    earliestRpc->originalFuture_->markCompleted(
        jitFuture.value(), jitFuture.dataPtrs());
  }
}

const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

std::shared_ptr<RpcAgent> RpcAgent::currentRpcAgent_ = nullptr;

bool RpcAgent::isCurrentRpcAgentSet() {
  return std::atomic_load(&currentRpcAgent_) != nullptr;
}

std::shared_ptr<RpcAgent> RpcAgent::getCurrentRpcAgent() {
  std::shared_ptr<RpcAgent> agent = std::atomic_load(&currentRpcAgent_);
  TORCH_INTERNAL_ASSERT(agent, "Current RPC agent is not set!");
  return agent;
}

void RpcAgent::setCurrentRpcAgent(std::shared_ptr<RpcAgent> rpcAgent) {
  if (rpcAgent) {
    std::shared_ptr<RpcAgent> previousAgent;
    // Use compare_exchange so that we don't actually perform the exchange if
    // that would trigger the assert just below. See:
    // https://en.cppreference.com/w/cpp/atomic/atomic_compare_exchange
    std::atomic_compare_exchange_strong(
        &currentRpcAgent_, &previousAgent, std::move(rpcAgent));
    TORCH_INTERNAL_ASSERT(
        previousAgent == nullptr, "Current RPC agent is set!");
  } else {
    // We can't use compare_exchange (we don't know what value to expect) but we
    // don't need to, as the only case that would trigger the assert is if we
    // replaced nullptr with nullptr, which we can just do as it has no effect.
    std::shared_ptr<RpcAgent> previousAgent =
        std::atomic_exchange(&currentRpcAgent_, std::move(rpcAgent));
    TORCH_INTERNAL_ASSERT(
        previousAgent != nullptr, "Current RPC agent is not set!");
  }
}

void RpcAgent::setTypeResolver(std::shared_ptr<TypeResolver> typeResolver) {
  typeResolver_ = std::move(typeResolver);
}

std::shared_ptr<TypeResolver> RpcAgent::getTypeResolver() {
  TORCH_INTERNAL_ASSERT(typeResolver_, "Type resolver is not set!");
  return typeResolver_;
}

void RpcAgent::enableGILProfiling(bool flag) {
  profilingEnabled_ = flag;
}

bool RpcAgent::isGILProfilingEnabled() {
  return profilingEnabled_.load();
}

DeviceMap RpcAgent::getDeviceMap(const WorkerInfo& /* unused */) const {
  // Default implementation has no device map.
  return {};
}

const std::vector<c10::Device>& RpcAgent::getDevices() const {
  // By default the agent is CPU-only.
  static const std::vector<c10::Device> noDevices = {};
  return noDevices;
}

std::unordered_map<std::string, std::string> RpcAgent::getDebugInfo() {
  /* This would later include more info other than metrics for eg: may include
     stack traces for the threads owned by the agent */
  // Default implementation: return getMetrics().
  return getMetrics();
}

std::ostream& operator<<(std::ostream& os, const WorkerInfo& workerInfo) {
  return os << "WorkerInfo(id=" << workerInfo.id_
            << ", name=" << workerInfo.name_ << ")";
}

} // namespace rpc
} // namespace distributed
} // namespace torch
