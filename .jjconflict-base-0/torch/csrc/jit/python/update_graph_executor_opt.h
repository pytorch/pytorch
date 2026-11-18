#pragma once
#include <torch/csrc/Export.h>
namespace torch::jit {
TORCH_API void setGraphExecutorOptimize(bool o);
TORCH_API bool getGraphExecutorOptimize();
} // namespace torch::jit
