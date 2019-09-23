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
    const Message& request) {
  AutoGIL ag;
  auto pickledPythonUDF =
      py::bytes(request.payload().data(), request.payload().size());
  TORCH_CHECK(runUDFFunction_ != nullptr, "runUDFFunction_ is nullptr");
  py::object res = runUDFFunction_(pickledPythonUDF);
  const auto& resStr = static_cast<std::string>(serialize(res));
  std::vector<char> payload(resStr.begin(), resStr.end());
  return payload;
}

py::object PythonRpcHandler::runPythonUDF(const std::string& pickledPythonUDF) {
  AutoGIL ag;
  return runUDFFunction_(py::bytes(pickledPythonUDF));
}

std::string PythonRpcHandler::serialize(const py::object& obj) {
  AutoGIL ag;
  return static_cast<std::string>((py::bytes)serializeFunction_(obj));
}

py::object PythonRpcHandler::deserialize(const std::string& serializedObj) {
  AutoGIL ag;
  return loadResultFunction_(py::bytes(serializedObj));
}

py::object PythonRpcHandler::loadPythonUDFResult(const Message& message) {
  AutoGIL ag;
  auto pargs = py::bytes(message.payload().data(), message.payload().size());
  TORCH_CHECK(loadResultFunction_ != nullptr, "loadResultFunction_ is nullptr");
  return loadResultFunction_(pargs);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
