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

TORCH_API bool HasInplaceOp(
    std::shared_ptr<Graph>& graph,
    const AliasDb& alias_db);

} // namespace jit
} // namespace torch
