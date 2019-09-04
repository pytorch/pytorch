#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

// singleton class provides interface to  execute python UDF remote call
// and deserialize the returned results by running python function
// in internal_rpc_utilities
// the singleton object is constructed at first when RPC agent is
// constructed, where the python function in internal_rpc_utilities are imported
// only once
class PythonRpcHandler {
 public:
  static PythonRpcHandler& getInstance();
  // execute python UDF, result is pickled to binary string
  std::vector<char> generatePythonUDFResult(const Message& request);
  // returned python UDF result is pickled binary string, so run python
  // function to unpickle the python UDF result and return pyObject to user
  py::object loadPythonUDFResult(const Message& message);

 private:
  PythonRpcHandler();
  ~PythonRpcHandler() = default;

  PythonRpcHandler(const PythonRpcHandler&) = delete;
  PythonRpcHandler& operator=(const PythonRpcHandler&) = delete;
  PythonRpcHandler(PythonRpcHandler&&) = delete;
  PythonRpcHandler& operator=(PythonRpcHandler&&) = delete;

  py::object runUDFFunction_;
  py::object loadResultFunction_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
