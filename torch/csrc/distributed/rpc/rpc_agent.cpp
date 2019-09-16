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

} // namespace rpc
} // namespace distributed
} // namespace torch
