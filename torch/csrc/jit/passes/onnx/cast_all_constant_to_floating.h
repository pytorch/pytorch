#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch::jit {
// see .cpp for docs
TORCH_API void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph);
} // namespace torch::jit
