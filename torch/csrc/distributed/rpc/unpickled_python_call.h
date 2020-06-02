#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

// This class converts the content in a PythonCall into py::object. This is a
// helper class to make sure that all arguments deserialization is done before
// entering RequestCallbackImpl::processRpc(...), so that the deserialization
// related logic can be carried out in one spot instead of scattered in multiple
// places for different message types.
// NB: The reason for not consolidating class into PythonCall is because
// PythonCall is a libtorch type which should not depend on Python types.
class TORCH_API UnpickledPythonCall : public RpcCommandBase {
 public:
  explicit UnpickledPythonCall(const SerializedPyObj& serializedPyObj);

  // toMessage() method is not implemented, as objects of this class should
  // never be directly converted into a Message object.
  Message toMessageImpl() && override;
  py::object movePythonUdf() &&;

 private:
  py::object pythonUdf_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
