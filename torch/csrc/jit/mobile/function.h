#pragma once

#include <vector>

#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/code.h>

namespace torch {
namespace jit {
enum OpCode : uint8_t;
struct Instruction;
struct OperatorString;

namespace mobile {

class TORCH_API Function : public torch::jit::Function {
 public:
  explicit Function(c10::QualifiedName name);
  Function(
      c10::QualifiedName name,
      Code code,
      at::optional<c10::FunctionSchema> schema);
  void run(Stack& stack) override;
  at::IValue operator()(Stack& stack);
  void ensure_defined() override {}
  size_t num_inputs() const override;
  const c10::QualifiedName& qualname() const override;
  bool call(Stack&, c10::function_ref<void(const mobile::Code&)>) override;

  // NOTE: the APIs below is dangerous: if you call append_instruction with
  // dbg_handle and then call it without; then the dbg_handle will become
  // misaligned. Therefore only use ONE variant at time.
  void append_instruction(OpCode op, int X, int N, int64_t dbg_handle);
  void append_instruction(OpCode op, int X, int N);
  void append_operator(
      const std::string& name,
      const std::string& overload_name,
      const c10::optional<int>& num_specified_args);
  void append_constant(const c10::IValue& constant);
  void append_type(const c10::TypePtr& type);
  void append_function(mobile::Function& func);

  void set_register_size(size_t size);

  int64_t get_debug_handle(size_t pc) const;
  const Code& get_code() const;
  Code& get_code();

  torch::jit::Function& setSchema(c10::FunctionSchema schema) override;
  bool hasSchema() const;
  const c10::FunctionSchema& getSchema() const override;

  // Returns the debug handle corresponding to where the execution
  // is halted due to exception.
  // If no corresponding debug handle is found then -1 is returned.
  const std::vector<int64_t>& getExceptionDebugHandles() const;
  static Function& registerFunc(
      const std::string& qualified_name,
      const std::vector<Instruction>& instructions,
      const std::vector<c10::IValue>& constants,
      const std::vector<c10::TypePtr>& types,
      const size_t register_size);

  // if not initialize, initialize by loading operators.
  // return true of all op loaded, return false if some op is not found
  // in the current runtime. Then, the ops that did not found will be filled
  // in unsupported_op_names
  bool initialize_operators(bool should_check_operators);

 private:
  c10::QualifiedName name_;
  Code code_;
  at::optional<c10::FunctionSchema> schema_; // (byte-code version 4+)
};

c10::optional<std::function<void(Stack&)>> makeOperatorFunction(
    c10::OperatorName opname,
    c10::optional<int> num_specified_args);

TORCH_API std::string operator_str(const c10::OperatorName& opname);

} // namespace mobile
} // namespace jit
} // namespace torch
