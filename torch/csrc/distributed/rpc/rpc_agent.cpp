#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

RpcAgent::RpcAgent(std::string workerName, worker_id_t id, RequestCallback cb)
    : workerName_(std::move(workerName)), id_(id), cb_(std::move(cb)) {}

RpcAgent::~RpcAgent() = default;

}
}
}
