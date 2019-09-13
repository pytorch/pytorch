#include <torch/csrc/distributed/rpc/python_udf_call.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonUDFCall::PythonUDFCall(std::vector<char>&& pickledPayload)
    : pickledPayload_(std::move(pickledPayload)) {}

PythonUDFCall::PythonUDFCall(const std::vector<char>& pickledPayload)
    : pickledPayload_(pickledPayload) {}

Message PythonUDFCall::toMessage() {
  return Message(
      std::move(pickledPayload_),
      std::vector<torch::Tensor>(),
      MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonUDFCall> PythonUDFCall::fromMessage(
    const Message& message) {
  return std::unique_ptr<PythonUDFCall>(new PythonUDFCall(message.payload()));
}

const std::vector<char>& PythonUDFCall::pickledPayload() const {
  return pickledPayload_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
