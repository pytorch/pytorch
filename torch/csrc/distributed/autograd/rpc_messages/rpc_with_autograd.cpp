#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
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
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    rpc::DeviceMap deviceMap)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      autogradMetadata_(autogradMetadata),
      wrappedMessage_(std::move(wrappedMessage)),
      deviceMap_(std::move(deviceMap)) {
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
  tensors_ = wrappedMessage_->tensors();
  wrappedMessageType_ = wrappedMessage_->type();
}

RpcWithAutograd::RpcWithAutograd(
    worker_id_t fromWorkerId,
    MessageType messageType,
    const AutogradMetadata& autogradMetadata,
    std::unique_ptr<RpcCommandBase> wrappedRpc,
    MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    rpc::DeviceMap deviceMap)
    : fromWorkerId_(fromWorkerId),
      messageType_(messageType),
      autogradMetadata_(autogradMetadata),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      deviceMap_(std::move(deviceMap)) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  TORCH_INTERNAL_ASSERT(
      messageType_ == MessageType::FORWARD_AUTOGRAD_REQ ||
      messageType_ == MessageType::FORWARD_AUTOGRAD_RESP);
}

c10::intrusive_ptr<Message> RpcWithAutograd::toMessageImpl() && {
  auto messageId = wrappedMessage_->id();
  auto wrappedMessageType = wrappedMessage_->type();

  auto payload = std::move(*wrappedMessage_).movePayload();
  TORCH_INTERNAL_ASSERT(!payload.empty());

  // Convert deviceMap to c10::Dict for serialization.
  c10::Dict<std::string, std::string> deviceMap;
  for (const auto& mapEntry : deviceMap_) {
    deviceMap.insert(mapEntry.first.str(), mapEntry.second.str());
  }

  std::vector<at::IValue> ivalues{
      wrappedMessageType,
      autogradMetadata_.autogradContextId,
      autogradMetadata_.autogradMessageId,
      fromWorkerId_,
      deviceMap};

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> additionalPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  // We shouldn't have any tensors!
  TORCH_INTERNAL_ASSERT(tensorTable.empty());

  // This wraps additionalPayload into payload and takes care of resizing,
  // encoding.
  rpc::writeWrappedPayload(payload, additionalPayload);

  return c10::make_intrusive<Message>(
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
  auto tupleElements = rpc::readWrappedPayload(payload, message);

  // Gather all the fields.
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 5);
  MessageType wrappedMessageType =
      static_cast<MessageType>(tupleElements[0].toInt());
  AutogradMetadata autogradMetadata(
      tupleElements[1].toInt(), tupleElements[2].toInt());
  worker_id_t workerId = tupleElements[3].toInt();
  auto c10DeviceMap =
      tupleElements[4].to<c10::Dict<std::string, std::string>>();

  // Convert to regular map.
  rpc::DeviceMap deviceMap;
  for (const auto& mapEntry : c10DeviceMap) {
    deviceMap.insert({mapEntry.key(), mapEntry.value()});
  }

  // Create new message type and build wrapped RPC.
  auto wrappedMessage = c10::make_intrusive<Message>(
      std::move(payload), std::move(tensors), wrappedMessageType, messageId);

  std::unique_ptr<RpcCommandBase> wrappedRpc;
  if (originalMessageType == MessageType::FORWARD_AUTOGRAD_REQ) {
    wrappedRpc = deserializeRequest(*wrappedMessage);
  } else {
    wrappedRpc = deserializeResponse(*wrappedMessage, wrappedMessageType);
  }

  return std::make_unique<RpcWithAutograd>(
      workerId,
      originalMessageType,
      autogradMetadata,
      std::move(wrappedRpc),
      wrappedMessageType,
      wrappedMessage->tensors(),
      deviceMap);
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

void RpcWithAutograd::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
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

const rpc::DeviceMap& RpcWithAutograd::deviceMap() {
  return deviceMap_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
