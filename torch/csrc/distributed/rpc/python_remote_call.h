#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/jit/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API PythonRemoteCall final {
 public:
  PythonRemoteCall(
      std::string pickledPythonUDF,
      at::IValue retRRefId,
      at::IValue retForkId);

  inline const std::string& udf() const {
    return pickledPythonUDF_;
  }

  inline const at::IValue& retRRefId() const {
    return retRRefId_;
  }

  inline const at::IValue& retForkId() const {
    return retForkId_;
  }

  Message toMessage() const;
  static PythonRemoteCall fromMessage(const Message& message);

 private:
  const std::string pickledPythonUDF_;
  const at::IValue retRRefId_;
  const at::IValue retForkId_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
