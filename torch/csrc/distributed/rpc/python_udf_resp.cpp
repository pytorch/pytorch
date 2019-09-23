#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonUDFResp::PythonUDFResp(std::vector<char> pickledPayload)
    : pickledPayload_(std::move(pickledPayload)) {}

Message PythonUDFResp::toMessage() {
  return Message(
      std::move(pickledPayload_),
      std::vector<torch::Tensor>(),
      MessageType::PYTHON_RET);
}

std::unique_ptr<PythonUDFResp> PythonUDFResp::fromMessage(
    const Message& message) {
  return c10::guts::make_unique<PythonUDFResp>(message.payload());
}

const std::vector<char>& PythonUDFResp::pickledPayload() const {
  return pickledPayload_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
