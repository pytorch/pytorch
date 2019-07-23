#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

static std::shared_ptr<Operator> matchOperator(
    at::Symbol symbol, std::string str_schema);

// A buildin operator
class TORCH_API BuiltinOp final {
 public:
  BuiltinOp(std::shared_ptr<Operator> op,
            std::vector<at::IValue> args)
      : stack_(std::move(args)), op_(op) {}

  ~BuiltinOp() {}

  std::shared_ptr<Operator> op();
  std::vector<at::IValue>& stack();
  Message toMessage();
  static BuiltinOp fromMessage(Message message);

 private:
  std::vector<at::IValue> stack_;
  std::shared_ptr<Operator> op_;
};

}
}
}
