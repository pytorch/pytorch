#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// Return value of a builtin operator
class TORCH_API BuiltinRet final {
 public:
  explicit BuiltinRet(std::vector<at::IValue> values)
      : values_(std::move(values)) {}

  ~BuiltinRet() {}

  std::vector<at::IValue>& values();
  Message toMessage();
  static BuiltinRet fromMessage(Message message);

 private:
  std::vector<at::IValue> values_;
};

}
}
}
