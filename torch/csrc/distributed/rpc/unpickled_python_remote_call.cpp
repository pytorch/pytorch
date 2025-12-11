#include <torch/csrc/distributed/rpc/unpickled_python_remote_call.h>

namespace torch::distributed::rpc {

UnpickledPythonRemoteCall::UnpickledPythonRemoteCall(
    const SerializedPyObj& serializedPyObj,
    const at::IValue& rrefId,
    const at::IValue& forkId,
    bool isAsyncExecution)
    : UnpickledPythonCall(serializedPyObj, isAsyncExecution),
      rrefId_(RRefId::fromIValue(rrefId)),
      forkId_(ForkId::fromIValue(forkId)) {}

const RRefId& UnpickledPythonRemoteCall::rrefId() const {
  return rrefId_;
}

const ForkId& UnpickledPythonRemoteCall::forkId() const {
  return forkId_;
}

} // namespace torch::distributed::rpc
