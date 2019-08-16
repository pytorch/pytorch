#include <torch/csrc/distributed/rpc/PythonRpcHandler.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace {
  py::object module_;
  py::object runUDFFunction_;
  py::object loadResultFunction_;
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
  }

  std::vector<char> generatePythonUDFResult(
    const Message& request) {
    AutoGIL ag;
    auto pargs = py::bytes(request.payload().data(), request.payload().size());
    py::bytes pres = runUDFFunction_(pargs);
    const auto& presStr = static_cast<std::string>(pres);
    std::vector<char> payload(presStr.begin(), presStr.end());
    return payload;
  }

  py::object loadPythonUDFResult(const Message& message) {
    AutoGIL ag;
    auto pargs = py::bytes(message.payload().data(), message.payload().size());
    return loadResultFunction_(pargs);
  }
} // PythonRpcHandler


} // rpc
} // distributed
} // torch
