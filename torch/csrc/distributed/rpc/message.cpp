#include <torch/csrc/distributed/rpc/message.h>

namespace torch {
namespace distributed {
namespace rpc {

Message::Message() = default;

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type)
    : payload_(payload), tensors_(tensors), type_(type) {}

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type,
    int64_t id)
    : payload_(payload), tensors_(tensors), type_(type), id_(id) {}

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

const std::vector<char>& Message::payload() const {
  return payload_;
}

std::vector<torch::Tensor>& Message::tensors() {
  return tensors_;
}

const std::vector<torch::Tensor>& Message::tensors() const {
  return tensors_;
}

const MessageType& Message::type() const {
  return type_;
}

bool Message::isRequest() const {
  return MessageType::SCRIPT_CALL == type_ ||
      MessageType::PYTHON_CALL == type_ || MessageType::REMOTE_CALL == type_ ||
      MessageType::MESSAGE_WITH_AUTOGRAD_REQ == type_ ||
      MessageType::RREF_FETCH_CALL == type_ ||
      MessageType::RREF_USER_CREATE == type_ ||
      MessageType::RREF_USER_DELETE == type_;
}

bool Message::requiresResponse() const {
  return MessageType::SCRIPT_CALL == type_ ||
      MessageType::PYTHON_CALL == type_ ||
      MessageType::MESSAGE_WITH_AUTOGRAD_REQ == type_ ||
      MessageType::RREF_FETCH_CALL == type_;
}

bool Message::isResponse() const {
  return MessageType::SCRIPT_RET == type_ || MessageType::PYTHON_RET == type_ ||
      MessageType::RREF_FETCH_RET == type_ ||
      MessageType::MESSAGE_WITH_AUTOGRAD_RESP == type_;
}

bool Message::isShutdown() const {
  return MessageType::SHUTDOWN == type_;
}

int64_t Message::id() const {
  return id_;
}

void Message::setId(int64_t id) {
  id_ = id;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
