#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/runtime/operator.h>

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
  explicit SchemaInfo(c10::FunctionSchema schema) : FunctionSchema(schema) {}
  explicit SchemaInfo(const char* signature)
      : FunctionSchema(torch::jit::getOperatorForLiteral(signature)->schema()) {
  }
};
} // namespace utils
} // namespace torch
