#include <torch/csrc/distributed/rpc/python_call.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonCall::PythonCall(
    std::vector<char> pickledPayload,
    std::vector<torch::Tensor> tensors)
    : pickledPayload_(std::move(pickledPayload)),
      tensors_(std::move(tensors)) {}

Message PythonCall::toMessage() && {
  return Message(
      std::move(pickledPayload_),
      std::move(tensors_),
      MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonCall> PythonCall::fromMessage(const Message& message) {
  return std::make_unique<PythonCall>(message.payload(), message.tensors());
}

const std::vector<char>& PythonCall::pickledPayload() const {
  return pickledPayload_;
}

const std::vector<torch::Tensor>& PythonCall::tensors() const {
  return tensors_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
