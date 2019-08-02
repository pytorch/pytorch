#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/distributed/rpc/Message.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// Return value of a builtin operator
class TORCH_API ScriptRet final {
 public:
  explicit ScriptRet(std::vector<at::IValue>&& values);

  const std::vector<at::IValue>& values();
  Message toMessage();
  static ScriptRet fromMessage(const Message& message);

 private:
  const std::vector<at::IValue> values_;
};

}
}
}
