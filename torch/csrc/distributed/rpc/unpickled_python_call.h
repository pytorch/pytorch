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
  UnpickledPythonCall(
      const SerializedPyObj& serializedPyObj,
      bool isAsyncExecution);
  ~UnpickledPythonCall() override;

  // toMessage() method is not implemented, as objects of this class should
  // never be directly converted into a Message object.
  c10::intrusive_ptr<Message> toMessageImpl() && override;
  const py::object& pythonUdf() const;

  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

 private:
  py::object pythonUdf_;
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
