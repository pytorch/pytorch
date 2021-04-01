#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph);
void FuseSigridTransformsListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

void ReplaceWithCopy(std::shared_ptr<torch::jit::Graph>& graph);

bool HasInplaceOp(std::shared_ptr<Graph>& graph, const AliasDb& alias_db);

} // namespace jit
} // namespace torch
