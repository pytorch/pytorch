#include <torch/csrc/distributed/rpc/rpc_with_autograd.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr int kAutogradMessageSize = 17;

AutogradMetadata::AutogradMetadata(
    int64_t autogradContextId_,
    int64_t autogradMessageId_)
    : autogradContextId(autogradContextId_),
      autogradMessageId(autogradMessageId_) {}

RpcWithAutograd::RpcWithAutograd(
    MessageType messageType,
    const AutogradMetadata& autogradMetadata,
    std::unique_ptr<RpcCommandBase> wrappedRpc)
    : messageType_(messageType), autogradMetadata_(autogradMetadata) {
  TORCH_INTERNAL_ASSERT(wrappedRpc != nullptr, "wrappedRpc cannot be null!");
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::MESSAGE_WITH_AUTOGRAD_REQ ||
      messageType_ == MessageType::MESSAGE_WITH_AUTOGRAD_RESP);
  wrappedMessage_ = std::move(*wrappedRpc).toMessage();
  tensors_ = wrappedMessage_.tensors();
  wrappedMessageType_ = wrappedMessage_.type();
}

RpcWithAutograd::RpcWithAutograd(
    MessageType messageType,
    const AutogradMetadata& autogradMetadata,
    std::unique_ptr<RpcCommandBase> wrappedRpc,
    MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors)
    : messageType_(messageType),
      autogradMetadata_(autogradMetadata),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::MESSAGE_WITH_AUTOGRAD_REQ ||
      messageType_ == MessageType::MESSAGE_WITH_AUTOGRAD_RESP);
}

Message RpcWithAutograd::toMessage() && {
  auto messageId = wrappedMessage_.id();
  auto messageType = wrappedMessage_.type();

  auto payload = std::move(wrappedMessage_).movePayload();
  TORCH_INTERNAL_ASSERT(!payload.empty());

  // We append the message type (1 byte), autograd context id(8 bytes) and
  // autograd message id(8 bytes) to the original message in network byte order
  // (big endian).
  size_t writableIndex = payload.size();

  // Need 17 additional bytes.
  payload.resize(payload.size() + kAutogradMessageSize);

  // Add message type.
  payload[writableIndex++] = messageType;

  // Add autograd ids.
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(payload.data()) + writableIndex,
      &autogradMetadata_.autogradContextId,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
  writableIndex += sizeof(int64_t);
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(payload.data()) + writableIndex,
      &autogradMetadata_.autogradMessageId,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);

  return Message(
      std::move(payload), std::move(tensors_), messageType_, messageId);
}

std::unique_ptr<RpcWithAutograd> RpcWithAutograd::fromMessage(
    const Message& message) {
  MessageType originalMessageType = message.type();
  TORCH_INTERNAL_ASSERT(
      MessageType::MESSAGE_WITH_AUTOGRAD_REQ == originalMessageType ||
      MessageType::MESSAGE_WITH_AUTOGRAD_RESP == originalMessageType);

  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t messageId = message.id();
  // Decode message type, autograd context id and autograd message id.
  auto payload = message.payload();
  TORCH_INTERNAL_ASSERT(payload.size() > kAutogradMessageSize);

  int64_t autogradContextId, autogradMessageId;
  // autograd message id.
  size_t indexToRead = payload.size() - sizeof(int64_t);
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  torch::utils::THP_decodeInt64Buffer(
      &autogradMessageId,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);

  // autograd context id.
  indexToRead -= sizeof(int64_t);
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  torch::utils::THP_decodeInt64Buffer(
      &autogradContextId,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);

  // message type.
  indexToRead -= 1;
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  MessageType wrappedMessageType =
      static_cast<MessageType>(payload[indexToRead]);

  // Remove the autograd information.
  payload.resize(payload.size() - kAutogradMessageSize);

  // Create new message type and build wrapped RPC.
  Message wrappedMessage(
      std::move(payload), std::move(tensors), wrappedMessageType, messageId);

  std::unique_ptr<RpcCommandBase> wrappedRpc;
  if (originalMessageType == MessageType::MESSAGE_WITH_AUTOGRAD_REQ) {
    wrappedRpc = deserializeRequest(wrappedMessage);
  } else {
    wrappedRpc = deserializeResponse(wrappedMessage);
  }

  return c10::guts::make_unique<RpcWithAutograd>(
      originalMessageType,
      AutogradMetadata(autogradContextId, autogradMessageId),
      std::move(wrappedRpc),
      wrappedMessageType,
      std::move(tensors));
}

std::vector<torch::Tensor>& RpcWithAutograd::tensors() {
  return tensors_;
}

const AutogradMetadata& RpcWithAutograd::autogradMetadata() const {
  return autogradMetadata_;
}

RpcCommandBase& RpcWithAutograd::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

MessageType RpcWithAutograd::wrappedMessageType() const {
  return wrappedMessageType_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
