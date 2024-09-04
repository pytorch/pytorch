#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Remove ops that do nothing on the forward pass (like aten::detach).
// This pass is invoked as a part of freeze_module.
// This function also takes a set of custom ops to eliminate. All ops in this
// set must take their output as their first input, i.e. x = f(x, ...)
TORCH_API bool EliminateNoOps(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<c10::Symbol> custom_ops = {});

} // namespace torch::jit
