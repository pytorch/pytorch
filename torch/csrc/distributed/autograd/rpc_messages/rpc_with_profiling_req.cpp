#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <vector>

namespace torch::distributed::autograd {

constexpr auto kProfilingResponseElementExpectedSize = 3;

using rpc::RpcCommandBase;

// This constructor is called when creating the RpcWithProfilingReq on the
// client.
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,
    rpc::ProfilingId profilingKeyId)
    : messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      tensors_(wrappedMessage_->tensors()),
      profilerConfig_(std::move(profilerConfig)),
      profilingKeyId_(profilingKeyId) {
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_REQ,
      c10::str(
          "Incorrect message type, expected message type ",
          rpc::MessageType::RUN_WITH_PROFILING_REQ));
  wrappedMessageType_ = wrappedMessage_->type();
}

// this constructor is only called in fromMessage() which is called in
// deserializeRequest(). It is called when reconstructing the
// RpcWithProfilingReq on the remote end.
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,
    rpc::ProfilingId profilingKeyId)
    : messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profilerConfig_(std::move(profilerConfig)),
      profilingKeyId_(profilingKeyId) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cant be null");
}

rpc::MessageType RpcWithProfilingReq::wrappedMessageType() const {
  return wrappedMessageType_;
}

void RpcWithProfilingReq::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

c10::intrusive_ptr<rpc::Message> RpcWithProfilingReq::toMessageImpl() && {
  // save the original message ID and type before moving it.
  auto wrappedMsgId = wrappedMessage_->id();
  auto wrappedMsgType = wrappedMessage_->type();
  // destructively move the wrappedMessage and get the payload. Now the payload
  // of wrappedMessage won't be in a valid state.
  auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
  // The wrapped payload should not be empty
  TORCH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload should not be empty.");
  // Create the ivalues to send over. We need to send the original message type
  // and id, as well as some profiling metadata.
  std::vector<at::IValue> ivalues{
      wrappedMsgType, profilerConfig_.toIValue(), profilingKeyId_.toIValue()};
  // Pickle it into a char payload to be sent over the wire.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  // add the profiling payload to the wrapped payload
  rpc::writeWrappedPayload(wrappedPayload, profilingPayload);
  // Put the wrapped payload into a message to return.
  auto returnMsg = c10::make_intrusive<rpc::Message>(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);

  return returnMsg;
}

RpcCommandBase& RpcWithProfilingReq::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

torch::autograd::profiler::ProfilerConfig RpcWithProfilingReq::
    getProfilingConfig() const {
  return profilerConfig_;
}

const rpc::ProfilingId& RpcWithProfilingReq::getProfilingId() const {
  return profilingKeyId_;
}

std::unique_ptr<RpcWithProfilingReq> RpcWithProfilingReq::fromMessage(
    const rpc::Message& message) {
  rpc::MessageType origMsgType = message.type();
  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t msgId = message.id();
  auto payload = message.payload();
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // Ensure that we have the expected number of elements
  TORCH_INTERNAL_ASSERT(
      tupleElements.size() == kProfilingResponseElementExpectedSize,
      c10::str(
          "Expected payload of size ",
          kProfilingResponseElementExpectedSize,
          " but got ",
          tupleElements.size()));
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  // Create a config to be enabled on this node that is a replica of the
  // state on the requesting node.
  torch::autograd::profiler::ProfilerConfig cfg =
      torch::autograd::profiler::ProfilerConfig::fromIValue(tupleElements[1]);

  rpc::ProfilingId profilerId = rpc::ProfilingId::fromIValue(tupleElements[2]);

  // Create new message type and build wrapped RPC
  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  TORCH_INTERNAL_ASSERT(
      wrappedMessage->isRequest(),
      "Messages wrapped with profiling requests must be requests.");
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeRequest(*wrappedMessage);

  return std::make_unique<RpcWithProfilingReq>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(cfg),
      profilerId);
}
} // namespace torch::distributed::autograd
