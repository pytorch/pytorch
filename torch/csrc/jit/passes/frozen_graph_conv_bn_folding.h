#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph);

TORCH_API void FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph);

TORCH_API void FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
