#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::distributed::rpc::Message;

void addSendRpcBackward(
    DistAutogradContext& autograd_context,
    Message& message) {
  // Attach the appropriate autograd edges.
  auto& tensors = message.tensors();
  if (torch::autograd::compute_requires_grad(tensors)) {
    auto grad_fn = std::make_shared<SendRpcBackward>();
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(tensors));

    // Add the appropriate input metadata for the grad_fn.
    for (const auto& tensor : tensors) {
      grad_fn->add_input_metadata(tensor);
    }

    // Update the autograd context and the message.
    auto& autogradContainer = DistAutogradContainer::getInstance();
    // Generate a new message id.
    int64_t autograd_message_id = autogradContainer.newAutogradMessageId();

    // Set autograd metadata for the message.
    message.setAutogradMetadata(Message::AutogradMetadata(
        autograd_context.context_id(), autograd_message_id));

    // Record the send autograd function in our current context.
    autograd_context.addSendFunction(grad_fn, autograd_message_id);
  }
}

DistAutogradContext* addRecvRpcBackward(
    torch::distributed::rpc::Message& message) {
  TORCH_INTERNAL_ASSERT(message.hasAutogradMetadata());

  auto& tensors = message.tensors();
  if (torch::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>();
    for (auto& tensor : tensors) {
      torch::autograd::set_history(tensor, grad_fn);
    }

    // Now update the autograd context with the necessary information.
    auto& autogradContainer = DistAutogradContainer::getInstance();
    // Initialize autograd context if necessary.
    DistAutogradContext& autogradContext =
        autogradContainer.createContext(message.getAutogradContextId());
    autogradContext.addRecvFunction(grad_fn, message.getAutogradMessageId());
    return &autogradContext;
  }
  return nullptr;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
