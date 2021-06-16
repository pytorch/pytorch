#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing the response of a Python UDF over RPC.
class TORCH_API PythonResp final : public RpcCommandBase {
 public:
  explicit PythonResp(SerializedPyObj&& serializedPyObj);

  c10::intrusive_ptr<Message> toMessageImpl() && override;

  static std::unique_ptr<PythonResp> fromMessage(const Message& message);

  const SerializedPyObj& serializedPyObj() const;

 private:
  SerializedPyObj serializedPyObj_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
