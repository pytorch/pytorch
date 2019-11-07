#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerInfo::MAX_NAME_LEN;

RpcAgent::RpcAgent(
    WorkerInfo workerId,
    std::unique_ptr<RequestCallback> cb,
    std::chrono::milliseconds globalProcessTimeout)
    : workerInfo_(std::move(workerId)),
      cb_(std::move(cb)),
      globalProcessTimeout_(globalProcessTimeout) {}

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

} // namespace rpc
} // namespace distributed
} // namespace torch
