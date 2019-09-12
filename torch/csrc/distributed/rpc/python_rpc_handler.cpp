#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace {

py::object module_;
py::object runUDFFunction_;
py::object loadResultFunction_;
py::object serializeFunction_;

} // anonymous namespace

namespace PythonRpcHandler {

void init() {
  AutoGIL ag;
  if (module_ == nullptr) {
    module_ = py::module::import("torch.distributed.internal_rpc_utils");
  }
  if (runUDFFunction_ == nullptr) {
    runUDFFunction_ = module_.attr("run_python_udf_internal");
  }
  if (loadResultFunction_ == nullptr) {
    loadResultFunction_ = module_.attr("load_python_udf_result_internal");
  }
  if (serializeFunction_ == nullptr) {
    serializeFunction_ = module_.attr("serialize");
  }
}

std::vector<char> generatePythonUDFResult(
    const Message& message,
    worker_id_t dst) {
  AutoGIL ag;
  auto pickledPythonUDF =
      py::bytes(message.payload().data(), message.payload().size());
  py::object res = runUDFFunction_(pickledPythonUDF);
  const auto& presStr = static_cast<std::string>(serialize(res, dst));
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object runPythonUDF(const std::string& pickledPythonUDF) {
  AutoGIL ag;
  return runUDFFunction_(py::bytes(pickledPythonUDF));
}

std::string serialize(const py::object& obj, worker_id_t dst) {
  AutoGIL ag;
  return static_cast<std::string>((py::bytes)serializeFunction_(obj, dst));
}

py::object deserialize(const std::string& serializedObj) {
  AutoGIL ag;
  return loadResultFunction_(py::bytes(serializedObj));
}

py::object loadPythonUDFResult(const Message& message) {
  AutoGIL ag;
  auto pargs = py::bytes(message.payload().data(), message.payload().size());
  return loadResultFunction_(pargs);
}

} // namespace PythonRpcHandler

} // namespace rpc
} // namespace distributed
} // namespace torch
