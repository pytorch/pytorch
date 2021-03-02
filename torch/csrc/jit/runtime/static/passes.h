#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch
