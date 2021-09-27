#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void FuseInferenceOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph);
TORCH_API void FuseListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

// If outputs_are_immutable is set to false, don't replace the view ops that
// produce aliases of graph outputs with the copy version.
TORCH_API void ReplaceWithCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

TORCH_API void EnableStaticRuntimeLayerNorm(
    std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API void RemoveImmutableInputDictLookups(
    std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API bool HasInplaceOp(
    std::shared_ptr<Graph>& graph,
    const AliasDb& alias_db);

TORCH_API bool forwardHasOp(const Module& module, const char* op_name);

TORCH_API bool hasJumpOps(std::shared_ptr<Graph>& graph);

TORCH_API void FuseSignLog1P(std::shared_ptr<Graph>& graph);

TORCH_API void UseVariadicTupleUnpack(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
