#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

#ifdef USE_CUDA
TORCH_API void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph);
#else
TORCH_API void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph) {}
#endif

} // namespace jit
} // namespace torch
