#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch::jit {
// see .cpp for docs
TORCH_API void RemoveInplaceOps(const std::shared_ptr<Graph>& graph);

TORCH_API void ImplicitCastForBinaryInplaceOps(Block* block);
} // namespace torch::jit
