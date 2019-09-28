#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerId::MAX_NAME_LEN;

RpcAgent::RpcAgent(WorkerId workerId, std::unique_ptr<RequestCallback> cb)
    : workerId_(std::move(workerId)), cb_(std::move(cb)) {}

RpcAgent::~RpcAgent() = default;

const WorkerId& RpcAgent::getWorkerId() const {
  return workerId_;
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
  defaultRpcAgent_ = defaultRpcAgent;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
