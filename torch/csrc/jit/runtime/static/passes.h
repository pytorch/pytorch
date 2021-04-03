#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph);
TORCH_API void FuseSigridTransformsListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API void ReplaceWithCopy(std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API bool HasInplaceOp(std::shared_ptr<Graph>& graph, const AliasDb& alias_db);

} // namespace jit
} // namespace torch
