#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing calling a Python function over RPC.
class TORCH_API PythonCall final : public RpcCommandBase {
 public:
  PythonCall(SerializedPyObj&& serializedPyObj, bool isAsyncExecution);

  Message toMessageImpl() && override;

  static std::unique_ptr<PythonCall> fromMessage(const Message& message);

  const SerializedPyObj& serializedPyObj() const;

  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

 private:
  SerializedPyObj serializedPyObj_;
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
