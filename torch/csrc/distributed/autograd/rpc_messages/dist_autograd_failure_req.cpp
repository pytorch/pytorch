#include <torch/csrc/distributed/autograd/rpc_messages/dist_autograd_failure_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace autograd {

DistAutogradFailureReq::DistAutogradFailureReq(
    int64_t context_id,
    std::string errorMsg)
    : context_id_(context_id), errorMsg_(std::move(errorMsg)) {}

rpc::Message DistAutogradFailureReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  // add context_id and errorMsg
  ivalues.emplace_back(context_id_);
  ivalues.emplace_back(errorMsg_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return rpc::Message(
      std::move(payload),
      std::move(tensorTable),
      rpc::MessageType::DIST_AUTOGRAD_FAILURE_REQ);
}

std::unique_ptr<DistAutogradFailureReq> DistAutogradFailureReq::fromMessage(
    const rpc::Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &message.tensors());
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();

  TORCH_INTERNAL_ASSERT(tupleElements.size() == 2);

  // recover errorMsg
  std::string errorMsg = tupleElements.back().toString()->string();
  tupleElements.pop_back();

  // recover context_id
  int64_t context_id = tupleElements.back().toInt();
  tupleElements.pop_back();

  return std::make_unique<DistAutogradFailureReq>(context_id, errorMsg);
}

int64_t DistAutogradFailureReq::getContextId() {
  return context_id_;
}

std::string DistAutogradFailureReq::getErrorMsg() {
  return errorMsg_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
