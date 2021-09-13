#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>

#include <unordered_map>
#include <vector>
namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

/**
 * The approach for caching operator lambdas uses the c10::OperatorName
 * as 'key', and OperatorFunctionWithSchema as 'value'. In case an
 * entry in the cach was found, we still need to determine if the
 * cached function has the same number of arguments. If it's different,
 * we can't use the cached value, and we need to re-compute it from
 * scratch.
 *
 * The expectation is that most of the time, we won't find a mismatch
 * when doing a lookup.
 *
 */
struct OperatorFunctionWithSchema {
  std::function<void(Stack&)> fn;
  c10::optional<int> num_specified_args;

  C10_NODISCARD bool has_same_arg_num(
      const c10::optional<int>& other_num_args) const {
    return other_num_args == num_specified_args;
  }
};

class Function {
 public:
  using OperatorCacheType =
      std::unordered_map<c10::OperatorName, OperatorFunctionWithSchema>;

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
      int64_t model_version, /* TODO: T90339189 deprecate all v3 when v3 models
                                are removed */
      OperatorCacheType& operator_cache);
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

 private:
  c10::QualifiedName name_;
  std::shared_ptr<Code> code_;
  at::optional<c10::FunctionSchema> schema_; // (byte-code version 4+)
};

} // namespace mobile
} // namespace jit
} // namespace torch
