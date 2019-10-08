#include <torch/csrc/distributed/rpc/python_udf_call.h>
#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonUDFCall::PythonUDFCall(
    std::vector<char> pickledPayload,
    std::vector<torch::Tensor> tensors)
    : pickledPayload_(std::move(pickledPayload)),
      tensors_(std::move(tensors)) {}

Message PythonUDFCall::toMessage() && {
  return Message(
      std::move(pickledPayload_),
      std::move(tensors_),
      MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonUDFCall> PythonUDFCall::fromMessage(
    const Message& message) {
  return c10::guts::make_unique<PythonUDFCall>(
      message.payload(), message.tensors());
}

const std::vector<char>& PythonUDFCall::pickledPayload() const {
  return pickledPayload_;
}

const std::vector<torch::Tensor>& PythonUDFCall::tensors() const {
  return tensors_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
