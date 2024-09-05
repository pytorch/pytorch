#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// return true if graph is modified
TORCH_API bool RefineIntegerValues(const std::shared_ptr<Graph>& graph);

} // namespace torch::jit
