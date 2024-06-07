#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/irange.h>

namespace torch {
namespace distributed {
namespace autograd {

using rpc::Message;
using rpc::MessageType;
using torch::autograd::Variable;

PropagateGradientsReq::PropagateGradientsReq(
    const AutogradMetadata& autogradMetadata,
    std::vector<Variable> grads,
    bool retainGraph)
    : autogradMetadata_(autogradMetadata),
      grads_(std::move(grads)),
      retainGraph_(retainGraph) {}

c10::intrusive_ptr<Message> PropagateGradientsReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  // Add all the grad tensors.
  ivalues.reserve(grads_.size() + 3);
  for (const auto& grad : grads_) {
    ivalues.emplace_back(grad);
  }

  // Now add autograd metadata.
  ivalues.emplace_back(autogradMetadata_.autogradContextId);
  ivalues.emplace_back(autogradMetadata_.autogradMessageId);

  // Add retain graph.
  ivalues.emplace_back(retainGraph_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensorTable),
      MessageType::BACKWARD_AUTOGRAD_REQ);
}

std::unique_ptr<PropagateGradientsReq> PropagateGradientsReq::fromMessage(
    const Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  const auto& tupleElements = tuple.toTupleRef().elements();

  // Build PropagateGradientsReq.
  TORCH_INTERNAL_ASSERT(tupleElements.size() >= 3);

  // Retrieve retainGraph.
  bool retainGraph = tupleElements.back().toBool();

  // Build AutogradMetadata.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t autogradContextId, autogradMessageId;
  autogradMessageId = tupleElements[tupleElements.size() - 2].toInt();
  autogradContextId = tupleElements[tupleElements.size() - 3].toInt();

  AutogradMetadata autogradMetadata(autogradContextId, autogradMessageId);

  // Retrieve the gradient tensors.
  std::vector<Variable> grads(tupleElements.size() - 3);
  for (const auto i : c10::irange(tupleElements.size() - 3)) {
    grads[i] = tupleElements[i].toTensor();
  }

  return std::make_unique<PropagateGradientsReq>(
      autogradMetadata, grads, retainGraph);
}

const AutogradMetadata& PropagateGradientsReq::getAutogradMetadata() {
  return autogradMetadata_;
}

const std::vector<torch::autograd::Variable>& PropagateGradientsReq::
    getGrads() {
  return grads_;
}

bool PropagateGradientsReq::retainGraph() {
  return retainGraph_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
