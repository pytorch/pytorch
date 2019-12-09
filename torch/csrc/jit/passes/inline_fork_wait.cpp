#include <torch/csrc/jit/passes/inline_fork_wait.h>

namespace torch {
namespace jit {

void InlineForkWait(
    Block* b,
    std::unordered_map<Value*, Value*>& future_remap) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::fork) {
      WithInsertPoint insert_guard(n);
      auto graph = b->owningGraph();
      auto subgraph = n->g(attr::Subgraph);

      auto output = insertGraph(*graph, *subgraph, n->inputs());

      future_remap[n->output()] = output.at(0);
    } else if (n->kind() == aten::wait) {
      AT_ASSERT(n->inputs().size() == 1);
      AT_ASSERT(n->outputs().size() == 1);
      n->output()->replaceAllUsesWith(future_remap.at(n->input()));
    }

    for (auto sub_b : n->blocks()) {
      InlineForkWait(sub_b, future_remap);
    }
  }
}

void InlineForkWait(const std::shared_ptr<Graph>& graph) {
  std::unordered_map<Value*, Value*> future_remap;
  InlineForkWait(graph->block(), future_remap);
}

} // namespace jit
} // namespace torch
