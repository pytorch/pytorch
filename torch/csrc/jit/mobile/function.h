#pragma once

#include <unordered_map>
#include <vector>

#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <c10/util/Exception.h>

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

class Function;
class BytecodeFunction : public torch::jit::Function {
 public:
  BytecodeFunction(mobile::Function& function) : function_(function) {}

  bool isGraphFunction() const override {
    return false;
  }

  void run(Stack& stack) override {
    throw c10::NotImplementedError("", "");
  }

  void run(Stack&& stack) override {
    throw c10::NotImplementedError("", "");
  }

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override {
    throw c10::NotImplementedError("", "");
  }

  at::IValue operator()(
      std::vector<at::IValue> stack,
      const Kwargs& kwargs = Kwargs()) override {
    throw c10::NotImplementedError("", "");
  }

  const c10::QualifiedName& qualname() const override;

  const std::string& name() const override;

  // if this isn't yet defined, run its method_creator function
  void ensure_defined() override {
    throw c10::NotImplementedError("", "");
  }

  std::shared_ptr<Graph> graph() const override {
    throw c10::NotImplementedError("", "");
  }

  std::shared_ptr<Graph> optimized_graph() const override {
    throw c10::NotImplementedError("", "");
  }

  void clear_execution_info() override {
    throw c10::NotImplementedError("", "");
  }

  GraphExecutor& get_executor() override {
    throw c10::NotImplementedError("", "");
  }

  const c10::FunctionSchema& getSchema() const override {
    throw c10::NotImplementedError("", "");
  }

  size_t num_inputs() const override {
    throw c10::NotImplementedError("", "");
  }

  void check_single_output() override {
    throw c10::NotImplementedError("", "");
  }

  std::string pretty_print_schema() const override {
    throw c10::NotImplementedError("", "");
  }

  Function& setSchema(c10::FunctionSchema schema) override {
    throw c10::NotImplementedError("", "");
  }

  const Code& getCode() const;

 private:
  mobile::Function& function_;
};

class Function {
 public:
  TORCH_API Function(c10::QualifiedName name);
  TORCH_API bool run(Stack& stack) const;
  c10::IValue operator()(Stack& stack) const;
  const std::string& name() const;
  TORCH_API const c10::QualifiedName& qualname() const;
  void append_instruction(OpCode op, int X, int N, int64_t dbg_handle = -1);
  bool append_operator(
      const std::string& name,
      const std::string& overload_name,
      const c10::optional<int>& num_specified_args,
      int64_t model_version); /* TODO: T90339189 deprecate all v3 when v3 models
                                are removed */
  void append_constant(const c10::IValue& constant);
  void append_type(const c10::TypePtr& type);

  void set_register_size(size_t size);

  int64_t get_debug_handle(size_t pc) const;
  const std::shared_ptr<Code> get_code() const;

  void setSchema(c10::FunctionSchema schema);
  const at::optional<c10::FunctionSchema>& getSchema() const;

  // Returns the debug handle corresponding to where the execution
  // is halted due to exception.
  // If no corresponding debug handle is found then -1 is returned.
  int64_t getExceptionDebugHandle() const;

  BytecodeFunction& getBytecodeFunction() {
    return bytecodeFunction_;
  }

 private:
  c10::QualifiedName name_;
  std::shared_ptr<Code> code_;
  at::optional<c10::FunctionSchema> schema_; // (byte-code version 4+)
  BytecodeFunction bytecodeFunction_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
