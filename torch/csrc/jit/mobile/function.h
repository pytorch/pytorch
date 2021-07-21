#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <vector>
#include "ATen/core/operator_name.h"

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

/**
 * There are 2 approaches we can use:
 *
 * 1. Approach-1: Hash on OperatorName and num_args: In this case, we use
 *    OperatorInfoWithSchema as the key for looking up the hash map. In
 *    the future, if we have more parameters, we just add them to this
 *    struct.
 *
 * 2. Approach-2: Hash on just OperatorName, but check if the fetched
 *    value is consistent (num args) with the current value, and use
 *    the cached function pointer only if this is consistent. Create a
 *    new instance, if it isn't consistent. The expectation is that most
 *    of the time, we won't find a mismatch when doing a lookup.
 *
 * This diff implements approach-2, but I also tried approach-1.
 * It's easy to switch between the 2, so I don't have a preference.
 * It just seemed cleaner to hash on a simpler key and make a check
 * later since we don't expect collision in practice. Plus, in the
 * future it's easier to extend to more complex checks related to
 * versioning which may not be trivial to hash.
 *
 */
struct OperatorInfoWithSchema {
  c10::OperatorName opname;
  c10::optional<int> num_specified_args;

  bool operator==(const OperatorInfoWithSchema rhs) const {
    return rhs.opname == opname && rhs.num_specified_args == num_specified_args;
  }
};

struct OperatorFunctionWithSchema {
  std::function<void(Stack&)> fn;
  c10::optional<int> num_specified_args;
};

class Function {
 public:
  typedef std::unordered_map<c10::OperatorName, OperatorFunctionWithSchema>
      OperatorCacheType;

  Function(c10::QualifiedName name);
  bool run(Stack& stack) const;
  c10::IValue operator()(Stack& stack) const;
  const std::string& name() const;
  const c10::QualifiedName& qualname() const;
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

namespace std {
template <>
struct hash<::torch::jit::mobile::OperatorInfoWithSchema> {
  size_t operator()(
      const ::torch::jit::mobile::OperatorInfoWithSchema& x) const {
    size_t h = std::hash<::c10::OperatorName>()(x.opname);
    if (x.num_specified_args.has_value()) {
      h ^= x.num_specified_args.value();
    }
    return h;
  }
};
} // namespace std
