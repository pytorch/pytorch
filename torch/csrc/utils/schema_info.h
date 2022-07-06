#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <unordered_set>

namespace torch {
namespace utils {

/**
 * class SchemaInfo
 *
 * Subclass of FunctionSchema that publicizes argument value specific operator
 * behavior (mutation, aliasing, special cases, etc...)
 */

struct TORCH_API SchemaInfo : c10::FunctionSchema {
 public:
  explicit SchemaInfo(c10::FunctionSchema schema)
      : FunctionSchema(schema), updated_(false) {}
  explicit SchemaInfo(const char* signature)
      : FunctionSchema(torch::jit::getOperatorForLiteral(signature)->schema()),
        updated_(false) {}

  bool is_mutable();

  bool is_mutable(size_t index);

  bool is_mutable(c10::string_view name);

  bool areAliasing(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs,
      bool check_additional);

  void addArgumentValue(const std::string& name, const at::IValue& value);

  void addArgumentValues(
      const std::vector<c10::optional<at::IValue>>& value_list);

  void addArgumentValues(
      const std::unordered_map<std::string, at::IValue>& values);

 private:
  at::IValue flattenZeroDimIValue(const at::IValue& value) const;

  void generateAliasMaps();

  // Map of argument IValues
  std::unordered_map<std::string, at::IValue> value_map_;

  // Alias map of inputs with each other
  std::vector<std::unordered_set<size_t>> input_alias_map_;

  // Alias map of outputs to inputs
  std::vector<std::unordered_set<size_t>> output_alias_map_;

  bool updated_;
};
} // namespace utils
} // namespace torch
