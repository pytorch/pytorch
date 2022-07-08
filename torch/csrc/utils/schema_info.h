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

struct TORCH_API SchemaInfo {
 public:
  explicit SchemaInfo(c10::FunctionSchema schema)
      : schema_(std::move(schema)) {}
  explicit SchemaInfo(const char* signature)
      : schema_(std::move(torch::jit::parseSchema(signature))) {}

 private:
  c10::FunctionSchema schema_;
};
} // namespace utils
} // namespace torch
