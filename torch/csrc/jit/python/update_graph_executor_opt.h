#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
namespace torch {
namespace jit {
TORCH_API void setGraphExecutorOptimize(bool o);
TORCH_API bool getGraphExecutorOptimize();
} // namespace jit
} // namespace torch
