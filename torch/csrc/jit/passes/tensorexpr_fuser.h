#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

// Run TensorExpressions-based fuser.
TORCH_API void fuseTensorExprs(std::shared_ptr<Graph>& graph);

// Register TensorExpressions-based fuser in custom passes.
TORCH_API void registerTensorExprFuser();

TORCH_API void setTensorExprFuserEnabled(bool val);

} // namespace jit
} // namespace torch
