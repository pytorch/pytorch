#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerInfo::MAX_NAME_LEN;

RpcAgent::RpcAgent(WorkerInfo workerId, std::unique_ptr<RequestCallback> cb)
    : workerInfo_(std::move(workerId)), cb_(std::move(cb)) {}

RpcAgent::~RpcAgent() = default;

const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
