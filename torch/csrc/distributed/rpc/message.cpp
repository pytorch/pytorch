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
    int64_t id,
    const AutogradMetadata& autograd_metadata)
    : payload_(payload),
      tensors_(tensors),
      type_(type),
      id_(id),
      autograd_metadata_(autograd_metadata) {}

Message::Message(const Message& other) = default;

Message::Message(Message&& other) noexcept = default;

Message& Message::operator=(Message const& rhs) & {
  auto payload = rhs.payload_;
  auto tensors = rhs.tensors_;
  Message(
      std::move(payload),
      std::move(tensors),
      rhs.type_,
      rhs.id_,
      rhs.autograd_metadata_)
      .swap(*this);
  return *this;
}

Message& Message::operator=(Message&& rhs) & {
  Message(
      std::move(rhs.payload_),
      std::move(rhs.tensors_),
      rhs.type_,
      rhs.id_,
      rhs.autograd_metadata_)
      .swap(*this);
  return *this;
}

void Message::swap(Message& rhs) noexcept {
  std::swap(payload_, rhs.payload_);
  std::swap(tensors_, rhs.tensors_);
  std::swap(type_, rhs.type_);
  std::swap(id_, rhs.id_);
  std::swap(autograd_metadata_, rhs.autograd_metadata_);
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
  return MessageType::SCRIPT_CALL == type_
      || MessageType::PYTHON_CALL == type_;
}

bool Message::isResponse() const {
  return MessageType::SCRIPT_RET == type_
      || MessageType::PYTHON_RET == type_;
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

bool Message::hasAutogradMetadata() const {
  return (
      getAutogradContextId() != kInvalidAutogradId &&
      getAutogradMessageId() != kInvalidAutogradId);
}

void Message::setAutogradMetadata(const AutogradMetadata& autograd_metadata) {
  autograd_metadata_ = autograd_metadata;
}

void Message::AutogradMetadata::setAutogradContextId(
    int64_t autograd_context_id) {
  autograd_context_id_ = autograd_context_id;
}

void Message::AutogradMetadata::setAutogradMessageId(
    int64_t autograd_message_id) {
  autograd_message_id_ = autograd_message_id;
}

int64_t Message::getAutogradContextId() const {
  return autograd_metadata_.getAutogradContextId();
}

int64_t Message::getAutogradMessageId() const {
  return autograd_metadata_.getAutogradMessageId();
}

int64_t Message::AutogradMetadata::getAutogradContextId() const {
  return autograd_context_id_;
}

int64_t Message::AutogradMetadata::getAutogradMessageId() const {
  return autograd_message_id_;
}

Message::AutogradMetadata::AutogradMetadata()
    : autograd_context_id_(kInvalidAutogradId),
      autograd_message_id_(kInvalidAutogradId) {}

Message::AutogradMetadata::AutogradMetadata(
    int64_t autograd_context_id,
    int64_t autograd_message_id)
    : autograd_context_id_(autograd_context_id),
      autograd_message_id_(autograd_message_id) {}
}
}
}
