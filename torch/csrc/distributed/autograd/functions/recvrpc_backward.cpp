#include <ATen/core/functional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::Variable;
using torch::autograd::variable_list;

RecvRpcBackward::RecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    ContextPtr autogradContext,
    rpc::worker_id_t fromWorkerId,
    rpc::DeviceMap deviceMap)
    : autogradMetadata_(autogradMetadata),
      autogradContext_(std::move(autogradContext)),
      fromWorkerId_(fromWorkerId),
      deviceMap_(std::move(deviceMap)) {}

variable_list RecvRpcBackward::apply(variable_list&& grads) {
  std::vector<Variable> outputGrads;
  for (const auto i : c10::irange(grads.size())) {
    const auto& grad = grads[i];
    if (grad.defined()) {
      outputGrads.emplace_back(grad);
    } else {
      // Put in zeros for a tensor with no grad.
      outputGrads.emplace_back(input_metadata(i).zeros_like());
    }
  }

  auto sharedContext = autogradContext_.lock();
  TORCH_CHECK(
      sharedContext,
      c10::str(
          "Autograd context no longer valid! This usually ",
          "means the autograd context was cleaned up by a different thread due ",
          "to an error before RecvRcpBackward had a chance to run"));

  // Send the gradients over the wire and record the future in the autograd
  // context.
  PropagateGradientsReq gradCall(
      autogradMetadata_,
      outputGrads,
      sharedContext->retrieveGraphTask()->keep_graph_);

  // Send the gradients over to the appropriate node.
  auto rpcAgent = rpc::RpcAgent::getCurrentRpcAgent();
  auto jitFuture = rpcAgent->send(
      rpcAgent->getWorkerInfo(fromWorkerId_),
      std::move(gradCall).toMessage(),
      rpc::kUnsetRpcTimeout,
      deviceMap_);

  // Record the future in the context.
  sharedContext->addOutstandingRpc(jitFuture);

  // 'recv' function sends the gradients over the wire using RPC, it doesn't
  // need to return anything for any downstream autograd function.
  return variable_list();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
