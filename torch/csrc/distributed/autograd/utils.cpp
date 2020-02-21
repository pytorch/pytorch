#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::distributed::autograd::AutogradMetadata;
using torch::distributed::autograd::RpcWithAutograd;
using torch::distributed::rpc::FutureMessage;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::RpcCommandBase;
using torch::distributed::rpc::WorkerInfo;

void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors) {
  // Attach autograd information only for tensors requiring grad.
  std::vector<torch::Tensor> tensors_with_grad;
  std::copy_if(
      tensors.begin(),
      tensors.end(),
      std::back_inserter(tensors_with_grad),
      [](const torch::Tensor& t) { return t.requires_grad(); });

  // Attach the appropriate autograd edges.
  auto grad_fn = std::make_shared<SendRpcBackward>();
  grad_fn->set_next_edges(
      torch::autograd::collect_next_edges(tensors_with_grad));

  // Add the appropriate input metadata for the grad_fn.
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);
  }

  // Record the send autograd function in our current context.
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId) {
  // Initialize autograd context if necessary.
  auto& autogradContainer = DistAutogradContainer::getInstance();
  auto autogradContext =
      autogradContainer.getOrCreateContext(autogradMetadata.autogradContextId);

  if (!tensors.empty() && torch::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>(
        autogradMetadata, autogradContext, fromWorkerId);
    for (auto& tensor : tensors) {
      if (tensor.requires_grad()) {
        torch::autograd::set_history(tensor, grad_fn);
      }
    }

    // Now update the autograd context with the necessary information.
    autogradContext->addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }

  return autogradContext;
}

Message getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    torch::distributed::rpc::Message&& wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording) {
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // If there is no valid context and no tensor requires grads, send original
  // rpc message. otherwise, attach grad info and grad functions and send
  // rpcWithAutograd message.
  auto tensorsRequireGrad =
      torch::autograd::compute_requires_grad(wrappedRpcMsg.tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return std::move(wrappedRpcMsg);
  }

  // Retrieve the appropriate context to modify.
  auto autogradContext = autogradContainer.currentContext();

  // Wrap the original rpc with autograd information.
  AutogradMetadata autogradMetadata(
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg));

  if (tensorsRequireGrad) {
    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // Record the workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();
}

std::shared_ptr<FutureMessage> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    torch::distributed::rpc::Message&& wrappedRpcMsg,
    bool forceGradRecording,
    const std::shared_ptr<torch::autograd::profiler::RecordFunctionAsync>& rf) {
  auto msg = getMessageWithAutograd(
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording);

  if (rf != nullptr) {
    rf->exitScope();
  }
  auto fut = agent.send(dst, std::move(msg));
  if (rf != nullptr) {
    // Add a callback to
    // the future that captures the RecordFunctionAsync to persist it for the
    // lifetime of the future. When the future is completed, this will run the
    // end() callbacks associated with the RecordFunctionAsync, so that async
    // RPCs can be profiled correctly.
    fut->addCallback(
        [rf](
            const Message& /* unused */,
            const c10::optional<utils::FutureError>& /* unused */) {
          rf->end();
        });
  }
  return fut;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
