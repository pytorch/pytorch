#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <cstddef>

namespace torch {
namespace jit {

TORCH_API size_t CalculateNecessaryArgs(
    const std::vector<Argument>& schema_args,
    at::ArrayRef<Value*> actual_inputs);

} // namespace jit
} // namespace torch
