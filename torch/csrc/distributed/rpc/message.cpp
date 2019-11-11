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
  return MessageType::SCRIPT_CALL == type_ || // dist.rpc on builtin ops
      MessageType::PYTHON_CALL == type_ || // dist.rpc on Python UDFs
      MessageType::SCRIPT_REMOTE_CALL == type_ || // dist.remote on builtin ops
      MessageType::PYTHON_REMOTE_CALL == type_ || // dist.remote on Python UDFs
      // RRef related internal messages
      MessageType::SCRIPT_RREF_FETCH_CALL == type_ ||
      MessageType::PYTHON_RREF_FETCH_CALL == type_ ||
      MessageType::RREF_USER_DELETE == type_ ||
      MessageType::RREF_CHILD_ACCEPT == type_ ||
      MessageType::RREF_FORK_REQUEST == type_ ||
      // Autograd message
      MessageType::BACKWARD_AUTOGRAD_REQ == type_ ||
      MessageType::FORWARD_AUTOGRAD_REQ == type_ ||
      // Cleanup Autograd context request
      MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ == type_;
}

bool Message::isResponse() const {
  return MessageType::SCRIPT_RET == type_ || // ret of dist.rpc on builtin ops
      MessageType::PYTHON_RET == type_ || // ret of dist.rpc on Python UDFs
      MessageType::REMOTE_RET == type_ || // ret of dist.remote
      MessageType::SCRIPT_RREF_FETCH_RET == type_ || // ret on RRef::toHere()
      MessageType::PYTHON_RREF_FETCH_RET == type_ || // ret on RRef::toHere()
      MessageType::EXCEPTION == type_ || // propagate back exceptions
      MessageType::RREF_ACK == type_ || // ret of other types
      // Autograd response
      MessageType::BACKWARD_AUTOGRAD_RESP == type_ ||
      MessageType::FORWARD_AUTOGRAD_RESP == type_ ||
      // Cleanup autograd context response
      MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP == type_;
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

Message createException(const Message& request, const std::exception& e) {
  std::string exceptionMsg = e.what();
  return createException(request, exceptionMsg);
}

Message createException(
    const Message& request,
    const std::string& exceptionStr) {
  std::vector<char> payload(exceptionStr.begin(), exceptionStr.end());
  return Message(
      std::move(payload),
      std::vector<torch::Tensor>(),
      MessageType::EXCEPTION,
      request.id());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
