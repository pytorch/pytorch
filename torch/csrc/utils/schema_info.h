#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace utils {

enum TORCH_API SchemaArgType { input, output };

/**
 * struct SchemaArgument
 *
 * Structure used to represent arguments or returns for a schema.
 */
struct TORCH_API SchemaArgument {
  SchemaArgType type;
  int index;
};

/**
 * class SchemaInfo
 *
 * Class that publicizes operator behavior (mutation, aliasing, special cases,
 * etc...)
 */

class TORCH_API SchemaInfo {
 public:
  SchemaInfo(c10::FunctionSchema schema) : schema_(schema) {}

  SchemaInfo(const char* signature)
      : schema_(torch::jit::getOperatorForLiteral(signature)->schema()) {}

  bool hasSideEffects() const;

  bool isDeterministic() const;

  bool isMutating(int index) const;

  bool isMutating(c10::string_view name) const;

  bool areAliasing(const SchemaArgument& lhs, const SchemaArgument& rhs) const;

  void addArgumentValue(const std::string& name, const at::IValue& value);

  void addArgumentValues(
      const std::vector<c10::optional<at::IValue>>& value_list);

  void addArgumentValues(
      const std::unordered_map<std::string, at::IValue>& values);

 private:
  std::vector<c10::Argument> getCorrectList(SchemaArgType type) const;

  c10::FunctionSchema schema_;
  std::unordered_map<std::string, at::IValue> value_map_;
};
} // namespace utils
} // namespace torch
