#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <string>

namespace torch {
namespace jit {

TORCH_API ::c10::FunctionSchema parseSchema(const std::string& schema);

} // namespace jit
} // namespace torch
