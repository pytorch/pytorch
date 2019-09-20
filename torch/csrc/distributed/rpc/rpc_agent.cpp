#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr size_t WorkerInfo::MAX_NAME_LEN;
using namespace torch::distributed::autograd;

RpcAgent::RpcAgent(WorkerInfo workerInfo, RequestCallback cb)
    : workerInfo_(std::move(workerInfo)), cb_(std::move(cb)) {}

RpcAgent::~RpcAgent() = default;

const WorkerInfo& RpcAgent::getWorkerInfo() const {
  return workerInfo_;
}

std::shared_ptr<FutureMessage> RpcAgent::send(
    const WorkerInfo& to,
    Message&& message) {
  // Record appropriate autograd information before sending the message over the
  // wire.
  auto& autogradContainer = DistAutogradContainer::getInstance();
  if (autogradContainer.hasValidContext()) {
    // Attach the appropriate autograd edges to the tensors found in the
    // message.
    auto grad_fn = addSendRpcBackward(message.tensors());

    // Record the send function in our current context.
    auto& currentContext = autogradContainer.currentContext();
    currentContext.addSendFunction(grad_fn);
  }
  return sendImpl(to, std::forward<Message>(message));
}
} // namespace rpc
} // namespace distributed
} // namespace torch
