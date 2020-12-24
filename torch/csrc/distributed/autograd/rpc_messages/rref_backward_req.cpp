#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace autograd {

using rpc::Message;
using rpc::MessageType;

RRefBackwardReq::RRefBackwardReq(
    const rpc::RRefId& rrefId,
    int64_t autogradContextId,
    bool retainGraph)
    : rrefId_(rrefId),
      autogradContextId_(autogradContextId),
      retainGraph_(retainGraph) {}

Message RRefBackwardReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;

  // Add all the fields.
  ivalues.emplace_back(rrefId_.toIValue());
  ivalues.emplace_back(autogradContextId_);
  ivalues.emplace_back(retainGraph_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return Message(
      std::move(payload),
      std::move(tensorTable),
      MessageType::RREF_BACKWARD_REQ);
}

std::unique_ptr<RRefBackwardReq> RRefBackwardReq::fromMessage(
    const Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &message.tensors());
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();

  // Build RRefBackwardReq.
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 3);

  // Retrieve all fields.
  bool retainGraph = tupleElements[2].toBool();
  int64_t autogradContextId = tupleElements[1].toInt();
  rpc::RRefId rrefId = rpc::RRefId::fromIValue(tupleElements[0]);

  return std::make_unique<RRefBackwardReq>(
      rrefId, autogradContextId, retainGraph);
}

const rpc::RRefId& RRefBackwardReq::getRRefId() const {
  return rrefId_;
}

int64_t RRefBackwardReq::getAutogradContextId() const {
  return autogradContextId_;
}

bool RRefBackwardReq::retainGraph() const {
  return retainGraph_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
