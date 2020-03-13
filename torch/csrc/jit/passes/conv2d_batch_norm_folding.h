#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {
TORCH_API void FoldConvBatchNorm2dOfFrozenTracedModuleGraph(std::shared_ptr<Graph>& graph);
} // namespace jit
} // namespace torch
