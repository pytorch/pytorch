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
      rpcAgentRunning_(true) {
  rpcRetryThread_ = std::thread(&RpcAgent::retryExpiredRpcs, this);
}

RpcAgent::~RpcAgent() {
  cleanup();
}

void RpcAgent::cleanup() {
  if (!rpcAgentRunning_.exchange(false)) {
    return;
  }
  rpcRetryMapCV_.notify_one();
  rpcRetryThread_.join();
}

std::shared_ptr<FutureMessage> RpcAgent::sendWithRetries(
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

  auto originalFuture = std::make_shared<FutureMessage>();
  steady_clock_time_point newTime =
      computeNewRpcRetryTime(retryOptions, /* retryCount */ 0);
  // Making a copy of the message so it can be retried after the first send.
  Message msgCopy = message;
  auto fm = send(to, std::move(message));
  auto firstRetryRpc = std::make_shared<RpcRetryInfo>(
      to,
      std::move(msgCopy),
      originalFuture,
      /* retryCount */ 0,
      retryOptions);

  fm->addCallback([this, newTime, firstRetryRpc](
                      const rpc::Message& lambdaMessage,
                      const c10::optional<utils::FutureError>& futErr) {
    rpcRetryCallback(lambdaMessage, futErr, newTime, firstRetryRpc);
  });

  return originalFuture;
}

void RpcAgent::retryExpiredRpcs() {
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
      auto fm = send(earliestRpc->to_, std::move(msgCopy));
      futures.emplace_back(fm, earliestRpc);

      // A callback will be attached to all futures for the retries in this
      // list. Thus they will either be rescheduled for future retries or they
      // will be marked as complete. We can safely delete them from the retry
      // Map for the current timepoint.
      it = earliestRpcList.erase(it);
    }

    lock.unlock();
    // We attach callbacks to the futures outside of the lock to prevent
    // potential deadlocks.
    for (const auto& it : futures) {
      auto fm = it.first;
      auto earliestRpc = it.second;
      steady_clock_time_point newTime = computeNewRpcRetryTime(
          earliestRpc->options_, earliestRpc->retryCount_);
      earliestRpc->retryCount_++;

      fm->addCallback([this, newTime, earliestRpc](
                          const rpc::Message& message,
                          const c10::optional<utils::FutureError>& futErr) {
        rpcRetryCallback(message, futErr, newTime, earliestRpc);
      });
    }

    // If there are no more RPC's set to be retried at the current timepoint,
    // we can remove the corresponsing unordered_set from the retry map. We
    // must also clear the futures vector.
    {
      std::lock_guard<std::mutex> retryMapLock(rpcRetryMutex_);
      futures.clear();
      if (earliestRpcList.empty()) {
        rpcRetryMap_.erase(earliestTimeout);
      }
    }
  }
}

void RpcAgent::rpcRetryCallback(
    const rpc::Message& message,
    const c10::optional<utils::FutureError>& futErr,
    steady_clock_time_point newTime,
    std::shared_ptr<RpcRetryInfo> earliestRpc) {
  if (futErr) {
    // Adding one since we want to include the original send as well and not
    // just the retry count.
    LOG(INFO) << "Send try " << std::to_string(earliestRpc->retryCount_ + 1)
              << " failed";
    if (earliestRpc->retryCount_ < earliestRpc->options_.maxRetries) {
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
      earliestRpc->originalFuture_->setError(errorMessage);
    }
  } else {
    // This try succeeded, so we can make the original future as complete.
    earliestRpc->originalFuture_->markCompleted(message);
  }
}

const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

std::shared_ptr<RpcAgent> RpcAgent::currentRpcAgent_ = nullptr;

bool RpcAgent::isCurrentRpcAgentSet() {
  return currentRpcAgent_ != nullptr;
}

std::shared_ptr<RpcAgent> RpcAgent::getCurrentRpcAgent() {
  TORCH_INTERNAL_ASSERT(currentRpcAgent_, "Current RPC agent is not set!");
  return currentRpcAgent_;
}

void RpcAgent::setCurrentRpcAgent(std::shared_ptr<RpcAgent> rpcAgent) {
  if (rpcAgent) {
    TORCH_INTERNAL_ASSERT(!currentRpcAgent_, "Current RPC agent is set!");
  } else {
    TORCH_INTERNAL_ASSERT(currentRpcAgent_, "Current RPC agent is not set!");
  }
  currentRpcAgent_ = std::move(rpcAgent);
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

std::unordered_map<std::string, std::string> RpcAgent::getDebugInfo() {
  /* This would later include more info other than metrics for eg: may include
     stack traces for the threads owned by the agent */
  // Default implementation: return getMetrics().
  return getMetrics();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
