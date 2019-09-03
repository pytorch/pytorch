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

  std::vector<char> generatePythonUDFResult(const py::bytes& pickledPythonUDF) {
    AutoGIL ag;
    py::bytes pres = runUDFFunction_(pickledPythonUDF);
    const auto& presStr = static_cast<std::string>(pres);
    std::vector<char> payload(presStr.begin(), presStr.end());
    return payload;
  }

  py::object runPythonUDF(const py::bytes& pickledPythonUDF) {
    AutoGIL ag;
    return runUDFFunction_(pickledPythonUDF, false);
  }

  std::string serialize(py::object obj) {
    AutoGIL ag;
    return static_cast<std::string>((py::bytes)serializeFunction_(obj));
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
} // PythonRpcHandler


} // rpc
} // distributed
} // torch
