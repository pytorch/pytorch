#pragma once

#include <unordered_map>
#include <vector>

#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/mobile/executor.h>

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

class TORCH_API Function : public torch::jit::Function {
 public:
  Function(c10::QualifiedName name);
  void run(Stack& stack) override;
  void run(Stack&& stack) override;
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override;
  at::IValue operator()(Stack stack, const Kwargs& = {}) override;
  void ensure_defined() override {}
  Executor& get_executor() override;
  size_t num_inputs() const override;
  void check_single_output() override;
  std::string pretty_print_schema() const override;
  const std::string& name() const override;
  const c10::QualifiedName& qualname() const override;
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

  torch::jit::Function& setSchema(c10::FunctionSchema schema) override;
  bool hasSchema() const;
  const c10::FunctionSchema& getSchema() const override;

  // Returns the debug handle corresponding to where the execution
  // is halted due to exception.
  // If no corresponding debug handle is found then -1 is returned.
  int64_t getExceptionDebugHandle() const;

 private:
  c10::QualifiedName name_;
  std::shared_ptr<Code> code_;
  at::optional<c10::FunctionSchema> schema_; // (byte-code version 4+)
  EdgeExecutor executor_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
