#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>

/*
 * API for query node-compatibility in CudaCodeGen
 *
 * It is used in the optimization passes, where the graph is traversed and parts
 * that could be handled by CudaCodegen is partitioned and stuffed in
 * `attr::Subgraph` of `prim::CudaFusionGroup`.
 *
 * Logic right now is very simple. On top of device placement, we consider a
 * `Node` compatible when we have a parsing rule for it in our parser.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_CU_API bool isFusibleCudaFusionGroup(const Node* node);

// consider if `node` could be fused into `fusion`
TORCH_CUDA_CU_API bool isFusibleCudaFusionGroup(
    const Node* fusion,
    const Node* node);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
