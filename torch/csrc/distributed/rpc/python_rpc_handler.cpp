#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonRpcHandler::PythonRpcHandler() {
  AutoGIL ag;
  py::object module_ =
      py::module::import("torch.distributed.internal_rpc_utils");
  runUDFFunction_ = module_.attr("run_python_udf_internal");
  loadResultFunction_ = module_.attr("load_python_udf_result_internal");
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const Message& request,
    std::vector<torch::Tensor>* tensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(request.payload().data(), request.payload().size());
  py::list tensors;
  for (auto t : request.tensors()) {
    tensors.append(t);
  }
  TORCH_CHECK(runUDFFunction_ != nullptr, "runUDFFunction_ is nullptr");
  py::tuple pres = runUDFFunction_(pargs, tensors);
  const auto& presStr = pres[0].cast<std::string>();
  const auto& presList = pres[1].cast<std::vector<torch::Tensor>>();
  std::vector<char> payload(presStr.begin(), presStr.end());
  for (auto t : presList) {
    tensorTable->emplace_back(t);
  }
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(const Message& message) {
  AutoGIL ag;
  auto pargs = py::bytes(message.payload().data(), message.payload().size());
  TORCH_CHECK(loadResultFunction_ != nullptr, "loadResultFunction_ is nullptr");
  py::list tensors;
  for (auto t : message.tensors()) {
    tensors.append(t);
  }
  return loadResultFunction_(pargs, tensors);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
