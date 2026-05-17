#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {
struct Graph;

// Propagates Device type info throughout the given graph.
TORCH_API bool DeviceTypePropagation(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
