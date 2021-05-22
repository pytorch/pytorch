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

Message::Message(const Message& other) = default;

Message::Message(Message&& other) noexcept = default;

Message& Message::operator=(Message const& rhs) & {
  auto payload = rhs.payload_;
  auto tensors = rhs.tensors_;
  Message(std::move(payload), std::move(tensors), rhs.type_, rhs.id_)
      .swap(*this);
  return *this;
}

Message& Message::operator=(Message&& rhs) & {
  Message(std::move(rhs.payload_), std::move(rhs.tensors_), rhs.type_, rhs.id_)
      .swap(*this);
  return *this;
}

void Message::swap(Message& rhs) noexcept {
  std::swap(payload_, rhs.payload_);
  std::swap(tensors_, rhs.tensors_);
  std::swap(type_, rhs.type_);
  std::swap(id_, rhs.id_);
}

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

Message createExceptionResponse(const std::exception& e, int64_t id) {
  return createExceptionResponse(e.what(), id);
}

Message createExceptionResponse(const std::string& exceptionStr, int64_t id) {
  std::vector<char> payload(exceptionStr.begin(), exceptionStr.end());
  return Message(
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

} // namespace

} // namespace rpc
} // namespace distributed
} // namespace torch
