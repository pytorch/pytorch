#include <torch/csrc/distributed/rpc/python_resp.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonResp::PythonResp(SerializedPyObj&& serializedPyObj)
    : serializedPyObj_(std::move(serializedPyObj)) {}

c10::intrusive_ptr<Message> PythonResp::toMessageImpl() && {
  auto payload = std::vector<char>(
      serializedPyObj_.payload_.begin(), serializedPyObj_.payload_.end());
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(serializedPyObj_.tensors_),
      MessageType::PYTHON_RET);
}

std::unique_ptr<PythonResp> PythonResp::fromMessage(const Message& message) {
  std::string payload(message.payload().begin(), message.payload().end());
  std::vector<Tensor> tensors = message.tensors();
  SerializedPyObj serializedPyObj(std::move(payload), std::move(tensors));
  return std::make_unique<PythonResp>(std::move(serializedPyObj));
}

const SerializedPyObj& PythonResp::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
