#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void FoldConvBatchnorm(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
