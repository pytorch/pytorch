#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <fusion.h>

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

namespace nvfuser {

constexpr int kPwThreadX = 128;
constexpr int kFcdReductionThreadX = 128;
constexpr int kNonFcdReductionThreadX = 32;
constexpr int kNonFcdReductionThreadY = 32;

TORCH_CUDA_CU_API bool hasReductionNode(const torch::jit::Block* block);
TORCH_CUDA_CU_API bool isReductionToSizeNode(const torch::jit::Node* node);
TORCH_CUDA_CU_API bool isReductionNode(const torch::jit::Node* node);

TORCH_CUDA_CU_API bool hasNormalizationNode(const torch::jit::Block* block);
TORCH_CUDA_CU_API bool isNormalizationNode(const torch::jit::Node* node);

TORCH_CUDA_CU_API bool isElementWiseNode(const torch::jit::Node* node);

// returns whether or not a parsing function exists for the given node type.
TORCH_CUDA_CU_API bool isNodeParsible(const torch::jit::Node* node);
TORCH_CUDA_CU_API bool shouldProfileNode(const torch::jit::Node* node);

TORCH_CUDA_CU_API bool skipNodeKind(const std::string& symbol_str, bool flip);

void InsertProfileNodes(torch::jit::ProfilingRecord* pr);

// lowers PyTorch jit graph to `Fusion`.
TORCH_CUDA_CU_API std::unique_ptr<Fusion> parseJitIR(
    const std::shared_ptr<torch::jit::Graph>& graph);

} // namespace nvfuser
