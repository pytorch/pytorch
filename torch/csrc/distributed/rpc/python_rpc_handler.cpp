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

void PythonRpcHandler::cleanUp() {
  AutoGIL ag;
  if (!runUDFFunction_.is_none()) {
    runUDFFunction_.dec_ref();
    runUDFFunction_ = py::none();
  }
  if (!loadResultFunction_.is_none()) {
    loadResultFunction_.dec_ref();
    loadResultFunction_ = py::none();
  }
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& requestTensorTable,
    std::vector<torch::Tensor>& responseTensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  // runUDFFunction_ should be always called before RpcAgent.join() and thus
  // it should not be none
  TORCH_CHECK(!runUDFFunction_.is_none(), "runUDFFunction_ is none");
  py::tuple pres = runUDFFunction_(pargs, requestTensorTable);
  const auto& presStr = pres[0].cast<std::string>();
  responseTensorTable = pres[1].cast<std::vector<torch::Tensor>>();
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& tensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  py::object loadResFunc;
  // loadResultFunction_ will be cleaned up in RpcAgent.join()
  // but loadPythonUDFResult() could be called after RpcAgent.join(), in this
  // rare case, we can import loadResFunc locally
  if (loadResultFunction_.is_none()) {
    loadResFunc = py::module::import("torch.distributed.internal_rpc_utils")
                      .attr("load_python_udf_result_internal");
  } else {
    loadResFunc = loadResultFunction_;
  }
  return loadResFunc(pargs, tensorTable);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
