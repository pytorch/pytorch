#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace autograd {

using rpc::Message;
using rpc::MessageType;
using rpc::RpcCommandBase;
using rpc::worker_id_t;

RpcWithAutograd::RpcWithAutograd(
    worker_id_t fromWorkerId,
    MessageType messageType,
    const AutogradMetadata& autogradMetadata,
    rpc::Message&& wrappedMessage)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      autogradMetadata_(autogradMetadata),
      wrappedMessage_(std::move(wrappedMessage)) {
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
  tensors_ = wrappedMessage_.tensors();
  wrappedMessageType_ = wrappedMessage_.type();
}

RpcWithAutograd::RpcWithAutograd(
    worker_id_t fromWorkerId,
    MessageType messageType,
    const AutogradMetadata& autogradMetadata,
    std::unique_ptr<RpcCommandBase> wrappedRpc,
    MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      autogradMetadata_(autogradMetadata),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
}

Message RpcWithAutograd::toMessage() && {
  rpc::isInRpcCall = true;
  auto messageId = wrappedMessage_.id();
  auto messageType = wrappedMessage_.type();

  auto payload = std::move(wrappedMessage_).movePayload();
  TORCH_INTERNAL_ASSERT(!payload.empty());

  std::vector<at::IValue> ivalues{messageType,
                                  autogradMetadata_.autogradContextId,
                                  autogradMetadata_.autogradMessageId,
                                  fromWorkerId_};

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> additionalPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  rpc::isInRpcCall = false;

  // We shouldn't have any tensors!
  TORCH_INTERNAL_ASSERT(tensorTable.empty());

  // Append the payload.
  payload.insert(
      payload.end(), additionalPayload.begin(), additionalPayload.end());

  // Add size of the additional payload.
  int64_t indexToWrite = payload.size();
  payload.resize(payload.size() + sizeof(int64_t));
  const int64_t additionalPayloadSize = additionalPayload.size();
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(payload.data()) + indexToWrite,
      &additionalPayloadSize,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);

  return Message(
      std::move(payload), std::move(tensors_), messageType_, messageId);
}

std::unique_ptr<RpcWithAutograd> RpcWithAutograd::fromMessage(
    const Message& message) {
  MessageType originalMessageType = message.type();
  TORCH_INTERNAL_ASSERT(
      MessageType::FORWARD_AUTOGRAD_REQ == originalMessageType ||
      MessageType::FORWARD_AUTOGRAD_RESP == originalMessageType);

  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t messageId = message.id();
  // Decode message type, autograd context id, autograd message id and worker
  // id from which we received this message.
  auto payload = message.payload();

  // Read the autograd payload remove it from the payload.
  int64_t autogradPayLoadSize;
  size_t indexToRead = payload.size() - sizeof(int64_t);
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  torch::utils::THP_decodeInt64Buffer(
      &autogradPayLoadSize,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
  payload.resize(indexToRead);

  // Now read the entire autograd payload and unpickle.
  TORCH_INTERNAL_ASSERT(payload.size() > autogradPayLoadSize)
  auto autogradPayLoadBegin =
      static_cast<const char*>(message.payload().data()) + payload.size() -
      autogradPayLoadSize;
  std::vector<torch::Tensor> tensorTable;
  IValue tuple = jit::unpickle(
      autogradPayLoadBegin,
      autogradPayLoadSize,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &tensorTable);
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();

  // Gather all the fields.
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 4);
  MessageType wrappedMessageType =
      static_cast<MessageType>(tupleElements[0].toInt());
  AutogradMetadata autogradMetadata(
      tupleElements[1].toInt(), tupleElements[2].toInt());
  worker_id_t workerId = tupleElements[3].toInt();
  payload.resize(payload.size() - autogradPayLoadSize);

  // Create new message type and build wrapped RPC.
  Message wrappedMessage(
      std::move(payload), std::move(tensors), wrappedMessageType, messageId);

  std::unique_ptr<RpcCommandBase> wrappedRpc;
  if (originalMessageType == MessageType::FORWARD_AUTOGRAD_REQ) {
    wrappedRpc = deserializeRequest(wrappedMessage);
  } else {
    wrappedRpc = deserializeResponse(wrappedMessage, wrappedMessageType);
  }

  return std::make_unique<RpcWithAutograd>(
      workerId,
      originalMessageType,
      autogradMetadata,
      std::move(wrappedRpc),
      wrappedMessageType,
      wrappedMessage.tensors());
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

std::unique_ptr<RpcCommandBase> RpcWithAutograd::moveWrappedRpc() && {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return std::move(wrappedRpc_);
}

MessageType RpcWithAutograd::wrappedMessageType() const {
  return wrappedMessageType_;
}

rpc::worker_id_t RpcWithAutograd::fromWorkerId() const {
  return fromWorkerId_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
