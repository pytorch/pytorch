#include <torch/csrc/distributed/rpc/python_call.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonCall::PythonCall(SerializedPyObj&& serializedPyObj)
    : serializedPyObj_(std::move(serializedPyObj)) {}

Message PythonCall::toMessageImpl() && {
  auto payload = std::vector<char>(
      serializedPyObj_.payload_.begin(), serializedPyObj_.payload_.end());
  return Message(
      std::move(payload),
      std::move(serializedPyObj_.tensors_),
      MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonCall> PythonCall::fromMessage(const Message& message) {
  std::string payload(message.payload().begin(), message.payload().end());
  std::vector<Tensor> tensors = message.tensors();
  SerializedPyObj serializedPyObj(std::move(payload), std::move(tensors));
  return std::make_unique<PythonCall>(std::move(serializedPyObj));
}

const SerializedPyObj& PythonCall::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
