#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerInfo::MAX_NAME_LEN;

RpcAgent::RpcAgent(
    WorkerInfo workerId,
    std::unique_ptr<RequestCallback> cb,
    std::chrono::milliseconds rpcTimeout)
    : workerInfo_(std::move(workerId)),
      cb_(std::move(cb)),
      rpcTimeout_(rpcTimeout),
      profilingEnabled_(false) {}

RpcAgent::~RpcAgent() = default;

const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

std::shared_ptr<RpcAgent> RpcAgent::defaultRpcAgent_ = nullptr;

std::shared_ptr<RpcAgent> RpcAgent::getDefaultRpcAgent() {
  TORCH_INTERNAL_ASSERT(
      defaultRpcAgent_, "Default rpc agent is not initialized!");
  return defaultRpcAgent_;
}

void RpcAgent::setDefaultRpcAgent(std::shared_ptr<RpcAgent> defaultRpcAgent) {
  TORCH_INTERNAL_ASSERT(
      !defaultRpcAgent_, "Default rpc agent is already initialized!");
  defaultRpcAgent_ = std::move(defaultRpcAgent);
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
