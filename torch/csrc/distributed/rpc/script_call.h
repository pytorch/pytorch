#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <optional>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::Operator;

// A ScriptCall instance represents an invocation of a builtin operator for a
// TorchScript function. If it is a builtin operator, it
// contains a shared ptr to the `Operator` and a list of arguments.
// If it is a TorchScript function, it contains a non empty qualifiedName string
// to the TorchScript function schema name and a list of arguments.
class TORCH_API ScriptCall : public RpcCommandBase {
 public:
  // Constructor for builitin operator call.
  ScriptCall(std::shared_ptr<Operator> op, std::vector<at::IValue>&& stack);
  // Constructor for TorchScript function call.
  ScriptCall(
      const c10::QualifiedName& qualifiedName,
      std::vector<at::IValue>&& stack,
      const bool isAsyncExecution = false);

  bool hasOp() const;
  std::shared_ptr<Operator> op() const;
  bool hasQualifiedName() const;
  const c10::QualifiedName& qualifiedName() const;
  // return the argument stack of this builtin operator
  const std::vector<at::IValue>& stack() const;
  std::vector<at::IValue>& stackRef();
  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptCall> fromMessage(const Message& message);

  ~ScriptCall() override = default;

 protected:
  virtual void toIValues(std::vector<at::IValue>& ivalues) const;
  static std::unique_ptr<ScriptCall> fromIValues(
      std::vector<at::IValue>& ivalues);

 private:
  // Given an operator symbol and a string schema, return the matched operator.
  static std::shared_ptr<Operator> matchOperator(const std::string& str_schema);

  static const std::string BUILTIN_OP_NAMESPACE_;
  static const std::string ATEN_PREFIX_;

  // This field has value if this ScriptCall represents invocation of a builtin
  // operator.
  std::optional<std::shared_ptr<Operator>> op_;
  // This field has non empty string if this ScriptCall represents invocation of
  // an annotated torchscript function defined by users.
  std::optional<const c10::QualifiedName> qualifiedName_;
  std::vector<at::IValue> stack_;
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
