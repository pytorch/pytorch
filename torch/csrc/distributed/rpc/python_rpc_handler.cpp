#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonRpcHandler::PythonRpcHandler() {
  AutoGIL ag;
  py::object module =
      py::module::import("torch.distributed.internal_rpc_utils");
  runUDFFunction_ = module.attr("run_python_udf_internal");
  loadResultFunction_ = module.attr("load_python_udf_result_internal");
  serializeFunction_ = module.attr("serialize");
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const Message& request,
    std::vector<torch::Tensor>& tensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(request.payload().data(), request.payload().size());
  TORCH_CHECK(runUDFFunction_ != nullptr, "runUDFFunction_ is nullptr");
  py::tuple pres =
      serializeFunction_(runUDFFunction_(pargs, request.tensors()));
  const auto& presStr = pres[0].cast<std::string>();
  tensorTable = pres[1].cast<std::vector<torch::Tensor>>();
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object PythonRpcHandler::runPythonUDF(
    const SerializedPyObj& serializedObj) {
  AutoGIL ag;
  return runUDFFunction_(
      py::bytes(serializedObj.payload_),
      serializedObj.tensors_);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  AutoGIL ag;
  py::tuple t = serializeFunction_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(),
      t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(SerializedPyObj serializedObj) {
  AutoGIL ag;
  return loadResultFunction_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

py::object PythonRpcHandler::loadPythonUDFResult(const Message& message) {
  AutoGIL ag;
  auto pargs = py::bytes(message.payload().data(), message.payload().size());
  TORCH_CHECK(loadResultFunction_ != nullptr, "loadResultFunction_ is nullptr");
  return loadResultFunction_(pargs, message.tensors());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
