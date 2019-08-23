#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace distributed {
namespace rpc {

// Return value of a builtin operator or a TorchScript function.
class TORCH_API ScriptRet final {
 public:
  explicit ScriptRet(at::IValue&& values);

  const at::IValue& value();
  Message toMessage();
  static ScriptRet fromMessage(const Message& message);

 private:
  const at::IValue value_;
};

}
}
}
