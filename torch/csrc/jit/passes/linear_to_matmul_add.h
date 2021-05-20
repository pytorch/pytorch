#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void DecomposeLinearToMatmulAdd(
    std::shared_ptr<Graph>& graph,
    bool restrict_to_gpu = true);

} // namespace jit
} // namespace torch
