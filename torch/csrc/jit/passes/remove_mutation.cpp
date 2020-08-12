#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch {
namespace jit {

void RemoveListMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeListMutation();
}

void RemoveTensorMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeTensorMutation();
}

} // namespace jit
} // namespace torch
