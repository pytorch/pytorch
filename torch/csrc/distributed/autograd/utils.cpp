#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::distributed::rpc::AutogradMetadata;
using torch::distributed::rpc::FutureMessage;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::RpcCommandBase;
using torch::distributed::rpc::RpcWithAutograd;
using torch::distributed::rpc::WorkerInfo;

void addSendRpcBackward(
    DistAutogradContext& autogradContext,
    const torch::distributed::rpc::AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors) {
  // Attach the appropriate autograd edges.
  if (torch::autograd::compute_requires_grad(tensors)) {
    auto grad_fn = std::make_shared<SendRpcBackward>();
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(tensors));

    // Add the appropriate input metadata for the grad_fn.
    for (const auto& tensor : tensors) {
      grad_fn->add_input_metadata(tensor);
    }

    // Record the send autograd function in our current context.
    autogradContext.addSendFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }
}

DistAutogradContext* addRecvRpcBackward(
    const torch::distributed::rpc::AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors) {
  if (torch::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>();
    for (auto& tensor : tensors) {
      torch::autograd::set_history(tensor, grad_fn);
    }

    // Now update the autograd context with the necessary information.
    auto& autogradContainer = DistAutogradContainer::getInstance();
    // Initialize autograd context if necessary.
    DistAutogradContext& autogradContext = autogradContainer.getOrCreateContext(
        autogradMetadata.autogradContextId);
    autogradContext.addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
    return &autogradContext;
  }
  return nullptr;
}

std::shared_ptr<FutureMessage> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::unique_ptr<RpcCommandBase> rpcCommand) {
  auto& autogradContainer = DistAutogradContainer::getInstance();
  if (autogradContainer.hasValidContext()) {
    // Retrieve the appropriate context to modify.
    auto& autogradContext = autogradContainer.currentContext();

    // Wrap the original rpc with autograd information.
    AutogradMetadata autogradMetadata(
        autogradContext.context_id(), autogradContainer.newAutogradMessageId());
    RpcWithAutograd rpcWithAutograd(
        MessageType::MESSAGE_WITH_AUTOGRAD_REQ,
        autogradMetadata,
        std::move(rpcCommand));

    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd.tensors());

    return agent.send(dst, std::move(rpcWithAutograd).toMessage());
  } else {
    return agent.send(dst, std::move(*rpcCommand).toMessage());
  }
}

} // namespace autograd
} // namespace distributed
} // namespace torch
