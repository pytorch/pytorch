#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {

Message::Message() : type_(MessageType::UNKNOWN) {}

Message::Message(
    std::vector<char> meta,
    std::vector<torch::Tensor> tensors,
    MessageType type)
    : meta_(std::move(meta)),
      tensors_(std::move(tensors)),
      type_(type) {}

Message::Message(
    std::vector<char> meta,
    std::vector<torch::Tensor> tensors,
    MessageType type,
    int64_t id)
    : meta_(std::move(meta)),
      tensors_(std::move(tensors)),
      type_(type),
      id_(id) {}

Message::Message(const Message & other)
    : meta_(other.meta_),
      tensors_(other.tensors_),
      type_(other.type_),
      id_(other.id_) {}

Message& Message::operator=(Message const & rhs) & {
  Message(rhs.meta_, rhs.tensors_, rhs.type_, rhs.id_).swap(*this);
  return *this;
}

Message& Message::operator=(Message && rhs) & {
  Message(std::move(rhs.meta_),
          std::move(rhs.tensors_),
          std::move(rhs.type_),
          rhs.id_).swap(*this);
  return *this;
}

void Message::swap(Message & rhs) noexcept {
  std::swap(meta_, rhs.meta_);
  std::swap(tensors_, rhs.tensors_);
  std::swap(type_, rhs.type_);
  std::swap(id_, rhs.id_);
}

Message::~Message() {}

std::vector<char>& Message::meta() {
  return meta_;
}

std::vector<torch::Tensor>& Message::tensors() {
  return tensors_;
}

const MessageType& Message::type() {
  return type_;
}

bool Message::isOp() {
  return MessageType::BUILTIN_OP == type_
      || MessageType::PYTHON_UDF_OP == type_;
}

bool Message::isRet() {
  return MessageType::BUILTIN_RET == type_
      || MessageType::PYTHON_UDF_RET == type_;
}

bool Message::isShutdown() {
  return MessageType::SHUTDOWN == type_;
}

int64_t Message::id() {
  return id_;
}

void Message::setId(int64_t id) {
  id_ = id;
}


}
}
}
