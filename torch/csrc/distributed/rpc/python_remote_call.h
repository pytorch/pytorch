#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API PythonRemoteCall : public RpcCommandBase {
 public:
  PythonRemoteCall(
      SerializedPyObj&& serializedPyObj,
      at::IValue retRRefId,
      at::IValue retForkId,
      DeviceMap&& deviceMap,
      const bool isAsyncExecution);

  inline const SerializedPyObj& serializedPyObj() const {
    return serializedPyObj_;
  }

  inline const at::IValue& retRRefId() const {
    return retRRefId_;
  }

  inline const at::IValue& retForkId() const {
    return retForkId_;
  }

  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

  inline const DeviceMap& deviceMap() const {
    return deviceMap_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<PythonRemoteCall> fromMessage(const Message& message);

 private:
  SerializedPyObj serializedPyObj_;
  const at::IValue retRRefId_;
  const at::IValue retForkId_;
  DeviceMap deviceMap_;
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
