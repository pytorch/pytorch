#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/distributed/rpc/unpickled_python_call.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::distributed::rpc {

// This class converts the content in a PythonRemoteCall into py::object. This
// is a helper class to make sure that all arguments deserialization is done
// before entering RequestCallbackImpl::processRpc(...), so that the
// deserialization related logic can be carried out in one spot instead of
// scattered in multiple places for different message types.
// NB: The reason for not consolidating class into PythonRemoteCall is because
// PythonRemoteCall is a libtorch type which should not depend on Python types.
class TORCH_API UnpickledPythonRemoteCall final : public UnpickledPythonCall {
 public:
  explicit UnpickledPythonRemoteCall(
      const SerializedPyObj& serializedPyObj,
      const at::IValue& retRRefId,
      const at::IValue& retForkId,
      const bool isAsyncExecution);

  const RRefId& rrefId() const;
  const ForkId& forkId() const;

 private:
  RRefId rrefId_;
  ForkId forkId_;
};

} // namespace torch::distributed::rpc
