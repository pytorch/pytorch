#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

/*
 * This file handles Parsing PyTorch jit ir;
 *
 * It is used in two places:
 *   1. When partitioning PyTorch jit ir to create prim::CudaFusionGroup, each
 *      node is queried by `isNodeParsible` to determine whether the node could
 *      be handled by the fuser (whether a given PyTorch jit operator should be
 *      merged);
 *   2. lowering PyTorch jit ir to CUDA codegen ir.
 *      creates a `Fusion` by traversing a PyTorch jit graph.
 *
 * TODO: we could consider exposing API to allow custom registration of parsing
 * rules for a given PyTorch jit operator.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// returns whether or not a parsing function exists for the given node type.
TORCH_CUDA_API bool isNodeParsible(const Node* const node);

// lowers PyTorch jit graph to `Fusion`.
TORCH_CUDA_API void parseJitIR(
    std::shared_ptr<Graph>& graph,
    CudaKernel* cuda_kernel);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
