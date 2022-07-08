#pragma once

#include <torch/csrc/jit/frontend/function_schema_parser.h>

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
      : FunctionSchema(torch::jit::parseSchema(signature)) {}
};
} // namespace utils
} // namespace torch
