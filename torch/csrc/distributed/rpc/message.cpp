#include <torch/csrc/distributed/rpc/message.h>

namespace torch {
namespace distributed {
namespace rpc {

Message::Message() = default;

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type)
    : payload_(std::move(payload)), tensors_(std::move(tensors)), type_(type) {}

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type,
    int64_t id)
    : payload_(std::move(payload)),
      tensors_(std::move(tensors)),
      type_(type),
      id_(id) {}

std::vector<char>&& Message::movePayload() && {
  return std::move(payload_);
}

std::vector<char>& Message::payload() {
  return payload_;
}

const std::vector<char>& Message::payload() const {
  return payload_;
}

std::vector<torch::Tensor>&& Message::moveTensors() && {
  return std::move(tensors_);
}

std::vector<torch::Tensor>& Message::tensors() {
  return tensors_;
}

const std::vector<torch::Tensor>& Message::tensors() const {
  return tensors_;
}

MessageType Message::type() const {
  return type_;
}

bool Message::isRequest() const {
  return MessageTypeFlags::REQUEST_TYPE & type_;
}

bool Message::isResponse() const {
  return MessageTypeFlags::RESPONSE_TYPE & type_;
}

int64_t Message::id() const {
  return id_;
}

void Message::setId(int64_t id) {
  id_ = id;
}

std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> Message::getStorages()
    const {
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages;
  storages.reserve(tensors_.size());
  for (const auto& tensor : tensors_) {
    storages.emplace_back(tensor.storage().getWeakStorageImpl());
  }
  return storages;
}

OutgoingMessage::OutgoingMessage() = default;

OutgoingMessage::OutgoingMessage(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type)
    : message_(c10::make_intrusive<Message>(
        std::move(payload),
        std::move(tensors),
        type
      )) {}

OutgoingMessage::OutgoingMessage(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type,
    int64_t id)
    : message_(c10::make_intrusive<Message>(
        std::move(payload),
        std::move(tensors),
        type,
        id
      )) {}

std::vector<char>&& OutgoingMessage::movePayload() && {
  return std::move(*message_).movePayload();
}

std::vector<char>& OutgoingMessage::payload() {
  return message_->payload();
}

const std::vector<char>& OutgoingMessage::payload() const {
  return message_->payload();
}

std::vector<torch::Tensor>&& OutgoingMessage::moveTensors() && {
  return std::move(*message_).moveTensors();
}

std::vector<torch::Tensor>& OutgoingMessage::tensors() {
  return message_->tensors();
}

const std::vector<torch::Tensor>& OutgoingMessage::tensors() const {
  return message_->tensors();
}

MessageType OutgoingMessage::type() const {
  return message_->type();
}

bool OutgoingMessage::isRequest() const {
  return message_->isRequest();
}

bool OutgoingMessage::isResponse() const {
  return message_->isResponse();
}

int64_t OutgoingMessage::id() const {
  return message_->id();
}

void OutgoingMessage::setId(int64_t id) {
  message_->setId(id);
}

std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> OutgoingMessage::getStorages()
    const {
  return message_->getStorages();
}

c10::intrusive_ptr<OutgoingMessage> createExceptionResponse(
    const std::exception& e,
    int64_t id) {
  return createExceptionResponse(e.what(), id);
}

c10::intrusive_ptr<OutgoingMessage> createExceptionResponse(
    const std::string& exceptionStr,
    int64_t id) {
  std::vector<char> payload(exceptionStr.begin(), exceptionStr.end());
  return c10::make_intrusive<OutgoingMessage>(
      std::move(payload),
      std::vector<torch::Tensor>(),
      MessageType::EXCEPTION,
      id);
}

namespace {

// NB: need to call torch::class_ to register Message in the map returned by
// c10::getCustomClassTypeMap(). Otherwise, Message cannot be wrapped within
// an IValue.
// NB: add this line here instead of in rpc/init.cpp because 1) we have C++
// only tests that won't run rpc/init.cpp; 2) Message is not meant to be
// visible from Python.
static const auto message = torch::class_<Message>("rpc", "_Message");
static const auto outgoingMessage = torch::class_<OutgoingMessage>("rpc", "_OutgoingMessage");

} // namespace

} // namespace rpc
} // namespace distributed
} // namespace torch
