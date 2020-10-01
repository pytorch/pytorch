#include <torch/csrc/distributed/rpc/unpickled_python_remote_call.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {

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

} // namespace rpc
} // namespace distributed
} // namespace torch
