#pragma once

#include <torch/csrc/jit/ir.h>

#include <memory>

namespace torch {
namespace jit {
// see .cpp for docs
TORCH_API void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph);
} // namespace jit
} // namespace torch
