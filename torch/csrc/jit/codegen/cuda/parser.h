#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>

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

constexpr int kPwThreadX = 128;
constexpr int kFcdReductionThreadX = 128;
constexpr int kNonFcdReductionThreadX = 32;
constexpr int kNonFcdReductionThreadY = 32;

TORCH_CUDA_CU_API bool hasReductionNode(const Block* block);

TORCH_CUDA_CU_API bool isReductionNode(const Node* node);

// returns whether or not a parsing function exists for the given node type.
TORCH_CUDA_CU_API bool isNodeParsible(const Node* node);

// lowers PyTorch jit graph to `Fusion`.
TORCH_CUDA_CU_API std::unique_ptr<Fusion> parseJitIR(
    const std::shared_ptr<Graph>& graph);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
