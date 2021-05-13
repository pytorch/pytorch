#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace autograd {

CleanupAutogradContextReq::CleanupAutogradContextReq(int64_t context_id)
    : context_id_(context_id){};

int64_t CleanupAutogradContextReq::getContextId() {
  return context_id_;
}

rpc::Message CleanupAutogradContextReq::toMessageImpl() && {
  // pickle context_id using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(at::IValue(context_id_), &tensorTable);
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
  IValue ivalue_context_id = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());

  // convert ivalue to int and construct request
  int64_t context_id = ivalue_context_id.toInt();
  return std::make_unique<CleanupAutogradContextReq>(context_id);
}

} // namespace autograd
} // namespace distributed
} // namespace torch
