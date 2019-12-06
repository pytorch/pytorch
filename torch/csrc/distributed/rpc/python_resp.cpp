#include <torch/csrc/distributed/rpc/python_resp.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonResp::PythonResp(
    std::vector<char> pickledPayload,
    std::vector<torch::Tensor> tensors)
    : pickledPayload_(std::move(pickledPayload)),
      tensors_(std::move(tensors)) {}

Message PythonResp::toMessage() && {
  return Message(
      std::move(pickledPayload_), std::move(tensors_), MessageType::PYTHON_RET);
}

std::unique_ptr<PythonResp> PythonResp::fromMessage(const Message& message) {
  return c10::guts::make_unique<PythonResp>(
      message.payload(), message.tensors());
}

const std::vector<char>& PythonResp::pickledPayload() const {
  return pickledPayload_;
}

const std::vector<torch::Tensor>& PythonResp::tensors() const {
  return tensors_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
