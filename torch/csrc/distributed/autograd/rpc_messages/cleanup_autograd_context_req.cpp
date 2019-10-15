#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/jit/pickle.h>

namespace torch {
namespace distributed {
namespace autograd {

CleanupAutogradContextReq::CleanupAutogradContextReq(int64_t context_id)
    : context_id_(context_id){};

int64_t CleanupAutogradContextReq::getContextId() {
  return context_id_;
}

rpc::Message CleanupAutogradContextReq::toMessage() && {
  std::vector<at::IValue> ivalues;
  ivalues.push_back(context_id_);

  // pickle
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  return rpc::Message(
      std::move(payload),
      std::move(tensorTable),
      rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ);
}

std::unique_ptr<CleanupAutogradContextReq> CleanupAutogradContextReq::
    fromMessage(const rpc::Message& message) {
  // unpickle and get the context_id we need to clean up
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  IValue tuple =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();
  int64_t context_id = tupleElements[0].toInt();

  return std::unique_ptr<CleanupAutogradContextReq>(
      new CleanupAutogradContextReq(context_id));
}

} // namespace autograd
} // namespace distributed
} // namespace torch
