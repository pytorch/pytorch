#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

RpcAgent::RpcAgent(WorkerId workerId, RequestCallback cb)
    : workerId_(std::move(workerId)), cb_(std::move(cb)) {}

RpcAgent::~RpcAgent() = default;

const WorkerId& RpcAgent::getWorkerId() const {
  return workerId_;
}

}
}
}
