#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
TORCH_API void registerCudaFuseGraph();

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void CudaFuseGraph(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
