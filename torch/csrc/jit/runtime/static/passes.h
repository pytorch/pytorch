#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph);
void FuseSigridTransformsListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

void ReplaceWithCopy(std::shared_ptr<torch::jit::Graph>& graph);

void SplitOutPrecomputeOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch
