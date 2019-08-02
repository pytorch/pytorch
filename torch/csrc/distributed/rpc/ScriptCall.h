#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

#define BUILTIN_OP_NAMESPACE "torch.ops.aten."

// A ScriptCall instance represents an invocation of a builtin operator. It
// contains a shared ptr to the `Operator` and a list of arguments.
class TORCH_API ScriptCall final {
 public:
  //ScriptCall(std::string qualifiedName, std::vector<at::IValue>&& args);
  ScriptCall(std::shared_ptr<Operator> op, std::vector<at::IValue>&& args);

  //at::IValue operator()() const;

  std::shared_ptr<Operator> op() const;
  // return the argument stack of this builtin operator
  const std::vector<at::IValue>& stack() const;

  Message toMessage();
  static ScriptCall fromMessage(const Message& message);

 private:

  static std::shared_ptr<Operator> matchOperator(
      at::Symbol& symbol, const std::string& str_schema);

  c10::optional<std::shared_ptr<Operator>> op_;
  const std::vector<at::IValue> stack_;
};

}
}
}
