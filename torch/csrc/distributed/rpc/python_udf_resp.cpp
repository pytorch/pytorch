#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonUDFResp::PythonUDFResp(
    std::vector<char> pickledPayload,
    std::vector<torch::Tensor> tensors)
    : pickledPayload_(std::move(pickledPayload)),
      tensors_(std::move(tensors)) {}

Message PythonUDFResp::toMessage() && {
  return Message(
      std::move(pickledPayload_), std::move(tensors_), MessageType::PYTHON_RET);
}

std::unique_ptr<PythonUDFResp> PythonUDFResp::fromMessage(
    const Message& message) {
  return c10::guts::make_unique<PythonUDFResp>(
      message.payload(), message.tensors());
}

const std::vector<char>& PythonUDFResp::pickledPayload() const {
  return pickledPayload_;
}

const std::vector<torch::Tensor>& PythonUDFResp::tensors() const {
  return tensors_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
