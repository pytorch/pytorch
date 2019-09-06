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
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

py::bytes PythonRpcHandler::runUDFFunction(const py::object& pargs) {
  AutoGIL ag;
  TORCH_CHECK(runUDFFunction_ != nullptr, "runUDFFunction_ is nullptr");
  return runUDFFunction_(pargs);
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const Message& request) {
  auto pargs = py::bytes(request.payload().data(), request.payload().size());
  auto pres = runUDFFunction(pargs);
  const auto& presStr = static_cast<std::string>(pres);
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(const Message& message) {
  auto pargs = py::bytes(message.payload().data(), message.payload().size());
  AutoGIL ag;
  TORCH_CHECK(loadResultFunction_ != nullptr, "loadResultFunction_ is nullptr");
  return loadResultFunction_(pargs);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
