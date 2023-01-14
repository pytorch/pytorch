#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {
TORCH_API void FuseQuantizedAddRelu(std::shared_ptr<Graph>& graph);
} // namespace torch::jit
