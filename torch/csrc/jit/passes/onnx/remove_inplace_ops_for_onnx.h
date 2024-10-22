#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model);

} // namespace torch::jit
