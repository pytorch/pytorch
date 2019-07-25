#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// A buildin operator
class TORCH_API BuiltinOp final {
 public:
  BuiltinOp(std::shared_ptr<Operator> op, std::vector<at::IValue>&& args);
  ~BuiltinOp();

  std::shared_ptr<Operator> op();
  // return the argument stack of this builtin operator
  std::vector<at::IValue>& stack();
  Message toMessage();
  static BuiltinOp fromMessage(const Message& message);
  static std::shared_ptr<Operator> matchOperator(
      at::Symbol& symbol, const std::string& str_schema);

 private:
  std::shared_ptr<Operator> op_;
  std::vector<at::IValue> stack_;
};

}
}
}
