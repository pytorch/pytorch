#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Fuses Convolution -> Batchnorm into a single Convolution by
// folding batchnorm weights into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API bool FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph);

// Fuses Convolution -> Add/Sub into a single Convolution by
// folding add constant tensor into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API bool FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph);

// Fuses Convolution -> Mul/Div into a single Convolution by
// folding add constant tensor into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API bool FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
