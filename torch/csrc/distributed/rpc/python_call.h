#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing calling a Python function over RPC.
class TORCH_API PythonCall final : public RpcCommandBase {
 public:
  explicit PythonCall(SerializedPyObj&& serializedPyObj);

  Message toMessageImpl() && override;

  static std::unique_ptr<PythonCall> fromMessage(const Message& message);

  const SerializedPyObj& serializedPyObj() const;

 private:
  SerializedPyObj serializedPyObj_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
