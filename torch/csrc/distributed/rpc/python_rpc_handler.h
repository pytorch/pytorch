#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

// Singleton class provides interface to execute python UDF remote call
// and deserialize the returned results by running python function
// in internal_rpc_utilities.
// The singleton object is constructed at first when RPC agent is
// constructed, where the python function in
// torch/distributed/internal_rpc_utils.py are imported only once.
class PYBIND11_EXPORT PythonRpcHandler {
 public:
  static PythonRpcHandler& getInstance();
  // Execute python UDF, result is pickled to binary string
  std::vector<char> generatePythonUDFResult(const Message& request);
  // Returned python UDF result is pickled binary string, so run python
  // function to unpickle the python UDF result and return py::object to user
  py::object loadPythonUDFResult(const Message& message);
  // Run a pickled Python UDF and return the result py::object
  py::object runPythonUDF(const std::string& pickledPythonUDF);
  // Serialized a py::object into a string
  std::string serialize(const py::object& obj);
  // Deserialize a string into a py::object
  py::object deserialize(const std::string& serializedObj);

 private:
  PythonRpcHandler();
  ~PythonRpcHandler() = default;

  PythonRpcHandler(const PythonRpcHandler&) = delete;
  PythonRpcHandler& operator=(const PythonRpcHandler&) = delete;
  PythonRpcHandler(PythonRpcHandler&&) = delete;
  PythonRpcHandler& operator=(PythonRpcHandler&&) = delete;

  py::object runUDFFunction_;
  py::object loadResultFunction_;
  py::object serializeFunction_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
