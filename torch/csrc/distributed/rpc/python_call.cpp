#include <torch/csrc/distributed/rpc/python_call.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonCall::PythonCall(SerializedPyObj&& serializedPyObj, bool isAsyncExecution)
    : serializedPyObj_(std::move(serializedPyObj)),
      isAsyncExecution_(isAsyncExecution) {}

c10::intrusive_ptr<Message> PythonCall::toMessageImpl() && {
  std::vector<char> payload;
  payload.reserve(serializedPyObj_.payload_.length() + 1);
  payload.push_back(isAsyncExecution_ ? 1 : 0);
  payload.insert(
      payload.end(),
      serializedPyObj_.payload_.begin(),
      serializedPyObj_.payload_.end());

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(serializedPyObj_.tensors_),
      MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonCall> PythonCall::fromMessage(const Message& message) {
  TORCH_INTERNAL_ASSERT(
      !message.payload().empty(),
      "Failed to convert an RPC message to PythonCall, the payload should at "
      "least contain one byte indicating whether this is an async function, "
      "but got payload of size ",
      message.payload().size());
  const char& c = message.payload()[0];
  TORCH_INTERNAL_ASSERT(c == 0 || c == 1);
  bool isAsyncExecution = (c == 1);
  std::string payload(message.payload().begin() + 1, message.payload().end());
  std::vector<Tensor> tensors = message.tensors();
  SerializedPyObj serializedPyObj(std::move(payload), std::move(tensors));
  return std::make_unique<PythonCall>(
      std::move(serializedPyObj), isAsyncExecution);
}

const SerializedPyObj& PythonCall::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
