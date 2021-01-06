#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Fuses Convolution -> Batchnorm into a single Convolution by
// folding batchnorm weights into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API void FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
