#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::Operator;

// A ScriptCall instance represents an invocation of a builtin operator for a
// TorchScript function (not implemented yet). If it is a builtin operator, it
// contains a shared ptr to the `Operator` and a list of arguments.
class TORCH_API ScriptCall {
 public:
  ScriptCall(std::shared_ptr<Operator> op, std::vector<at::IValue>&& args);

  std::shared_ptr<Operator> op() const;
  // return the argument stack of this builtin operator
  const std::vector<at::IValue>& stack() const;
  std::vector<at::IValue>& stackRef();

  Message toMessage();
  static ScriptCall fromMessage(const Message& message);

  virtual ~ScriptCall() = default;

 protected:
  virtual void toIValues(std::vector<at::IValue>& ivalues) const;
  static std::shared_ptr<Operator> fromIValues(
      std::vector<at::IValue>& ivalues);

 private:
  // Given an operator symbol and a string schema, return the matched operator.
  static std::shared_ptr<Operator> matchOperator(const std::string& str_schema);

  static const std::string BUILTIN_OP_NAMESPACE_;
  static const std::string ATEN_PREFIX_;

  // This field has value if this ScriptCall represents invocation of a builtin
  // operator.
  c10::optional<std::shared_ptr<Operator>> op_;
  std::vector<at::IValue> stack_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
