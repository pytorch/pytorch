#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace rpc {
const std::string REMOTE_PROFILING_KEY_PREFIX = "#remote_op: ";
constexpr int kAutoIncrementBits = 48;
/*static */ thread_local c10::optional<std::string>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    RemoteProfilerManager::currentThreadLocalKey_ = c10::nullopt;
/*static */ RemoteProfilerManager& RemoteProfilerManager::getInstance() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static RemoteProfilerManager* handler = new RemoteProfilerManager();
  return *handler;
}

void RemoteProfilerManager::setCurrentKey(std::string key) {
  // We should not allow overriding the current key, it needs to be committed
  // with writeKey() explicitly first.
  if (RemoteProfilerManager::currentThreadLocalKey_) {
    TORCH_CHECK(
        false,
        "Cannot call RemoteProfilerManager::setCurrentKey when current key is already set.");
  }
  currentThreadLocalKey_ = std::move(key);
}

bool RemoteProfilerManager::isCurrentKeySet() const {
  return currentThreadLocalKey_ ? true : false;
}

void RemoteProfilerManager::unsetCurrentKey() {
  currentThreadLocalKey_ = c10::nullopt;
}

void RemoteProfilerManager::eraseKey(const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  profiledRpcKeys_.erase(it);
}

std::string RemoteProfilerManager::retrieveRPCProfilingKey(
    const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  return it->second;
}

ProfilingId RemoteProfilerManager::getNextProfilerId() {
  auto localId = getNextLocalId();
  auto localWorkerId = RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_;
  auto globallyUniqueId =
      torch::distributed::rpc::ProfilingId(localWorkerId, localId);
  return globallyUniqueId;
}

local_id_t RemoteProfilerManager::getNextLocalId() {
  std::lock_guard<std::mutex> guard(mutex_);
  return currentLocalId_++;
}

std::string& RemoteProfilerManager::getCurrentProfilingKey() {
  TORCH_CHECK(
      RemoteProfilerManager::currentThreadLocalKey_,
      "Must set currentThreadLocalKey_ before calling getCurrentProfilingKey");
  return *currentThreadLocalKey_;
}

void RemoteProfilerManager::saveRPCKey(
    ProfilingId globallyUniqueId,
    const std::string& rpcProfilingKey) {
  std::lock_guard<std::mutex> guard(mutex_);
  profiledRpcKeys_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(globallyUniqueId),
      std::forward_as_tuple(rpcProfilingKey));
}

RemoteProfilerManager::RemoteProfilerManager() {
  auto workerId =
      static_cast<int64_t>(RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_);
  currentLocalId_ = workerId << kAutoIncrementBits;
}
} // namespace rpc
} // namespace distributed
} // namespace torch
