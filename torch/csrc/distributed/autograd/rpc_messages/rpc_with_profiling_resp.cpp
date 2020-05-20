#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace autograd {
using rpc::RpcCommandBase;

// This constructor is called when creating the RpcProfilingResp before sending
// it as a message over the wire.
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::worker_id_t fromWorkerId,
    rpc::MessageType messageType,
    rpc::Message&& wrappedMessage,
    std::string profiledEvents,
    std::string profilingKey)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      profiledEvents_(std::move(profiledEvents)),
      profilingKey_(std::move(profilingKey)) {
  tensors_ = wrappedMessage_.tensors();
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_RESP,
      "Incorrect Message type");
  wrappedMessageType_ = wrappedMessage_.type();
}
// this constructor is called in fromMessage() which is called when
// reconstructing this RPC command when processing a message of this type
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::worker_id_t fromWorkerId,
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    std::string profiledEvents,
    std::string profilingKey)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profiledEvents_(std::move(profiledEvents)),
      profilingKey_(std::move(profilingKey)) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrapped RPC cannot be null");
}

std::unique_ptr<RpcCommandBase> RpcWithProfilingResp::moveWrappedRpc() && {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return std::move(wrappedRpc_);
}

rpc::MessageType RpcWithProfilingResp::wrappedMessageType() const {
  return wrappedMessageType_;
}

std::string RpcWithProfilingResp::getProfilingKey() const {
  return profilingKey_;
}

void RpcWithProfilingResp::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

rpc::Message RpcWithProfilingResp::toMessageImpl() && {
  auto wrappedMsgId = wrappedMessage_.id();
  auto wrappedMsgType = wrappedMessage_.type();
  auto wrappedPayload = std::move(wrappedMessage_).movePayload();
  // Wrapped payload should not be empty
  TORCH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload cannot be empty");
  // Create ivalues to send over
  std::vector<at::IValue> ivalues{
      wrappedMsgType, profiledEvents_, profilingKey_, fromWorkerId_};
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  rpc::generateWrappedPayload(wrappedPayload, profilingPayload);
  auto returnMsg = rpc::Message(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);
  return returnMsg;
}

RpcCommandBase& RpcWithProfilingResp::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

std::string RpcWithProfilingResp::getProfiledEvents() const {
  return profiledEvents_;
}

std::unique_ptr<RpcWithProfilingResp> RpcWithProfilingResp::fromMessage(
    const rpc::Message& message) {
  rpc::MessageType origMsgType = message.type();
  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t msgId = message.id();
  auto payload = message.payload();
  auto tupleElements = rpc::readPayload(payload, message);
  // Ensure that we have the expected number of elements
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 4);
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  std::string profiledEvents = tupleElements[1].toStringRef();
  std::string profilingKey = tupleElements[2].toStringRef();
  int fromWorkerId = tupleElements[3].toInt();
  rpc::Message wrappedMessage(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  TORCH_INTERNAL_ASSERT(
      wrappedMessage.isResponse(),
      "Messages wrapped with profiling response must be responses.");
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeResponse(wrappedMessage, wrappedMsgType);
  return std::make_unique<RpcWithProfilingResp>(
      fromWorkerId,
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      wrappedMessage.tensors(),
      std::move(profiledEvents),
      std::move(profilingKey));
}

} // namespace autograd

} // namespace distributed
} // namespace torch
