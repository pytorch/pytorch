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

  // Deserialize Python function, run it, and serialize its return value.
  std::vector<char> generatePythonUDFResult(
      const std::vector<char>& pickledPayload,
      const std::vector<torch::Tensor>& requestTensorTable,
      std::vector<torch::Tensor>& responseTensorTable);

  // Returned python UDF result is pickled binary string, so run python
  // function to unpickle the python UDF result and return py::object to user
  py::object loadPythonUDFResult(
      const std::vector<char>& pickledPayload,
      const std::vector<torch::Tensor>& tensorTable);

  // Run a pickled Python UDF and return the result py::object
  py::object runPythonUDF(const SerializedPyObj& serializedObj);

  // Serialized a py::object into a string
  SerializedPyObj serialize(const py::object& obj);

  // Deserialize a string into a py::object
  py::object deserialize(const SerializedPyObj& serializedObj);

  // Check if obj is RemoteException, then throw it
  void handleException(const py::object& obj);

  // Explicitly clean up py::objects to avoid segment faults when
  // py::objects with CPython are cleaned up later at program exit
  // See similar issues reported https://github.com/pybind/pybind11/issues/1598
  // and https://github.com/pybind/pybind11/issues/1493
  // Our local tests also caught this segment faults if py::objects are cleaned
  // up at program exit. The explaination is: CPython cleans up most critical
  // utilities before cleaning up PythonRpcHandler singleton, so when
  // PythonRpcHandler signleton cleans up py::objects and call dec_ref(), it
  // will crash.
  // The solution is to clean up py::objects earlier when Rpc agent join().
  // Be note that py::objects can not be cleaned up when Rpc agent is destroyed
  // as well, as Rpc agent is global variable and it will have same issue as
  // PythonRpcHandler.
  void cleanup();

 private:
  PythonRpcHandler();
  ~PythonRpcHandler() = default;

  PythonRpcHandler(const PythonRpcHandler&) = delete;
  PythonRpcHandler& operator=(const PythonRpcHandler&) = delete;
  PythonRpcHandler(PythonRpcHandler&&) = delete;
  PythonRpcHandler& operator=(PythonRpcHandler&&) = delete;

  // Ref to `torch.distributed.rpc.internal._run_function`.
  py::object pyRunFunction_;

  // Ref to `torch.distributed.rpc.internal._load_return_value`.
  py::object pyLoadReturnValue_;

  // Ref to `torch.distributed.rpc.internal.serialize`.
  py::object pySerialize_;

  // Ref to 'torch.distributed.rpc.internal._handle_exception'
  py::object pyHandleException_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
