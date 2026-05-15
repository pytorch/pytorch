#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>

namespace torch::jit {

static void InlineForkWait(
    Block* b,
    std::unordered_map<Value*, Value*>& future_remap) {
  auto nodes = b->nodes();

  // Track the futures returned by prim::fork.
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    if (node->kind() != prim::fork) {
      continue;
    }
    WithInsertPoint insert_guard(node);
    auto graph = b->owningGraph();
    auto subgraph = node->g(attr::Subgraph);

    auto output = insertGraph(*graph, *subgraph, node->inputs());

    future_remap[node->output()] = output.at(0);
  }

  // Remove aten::wait if its input future is returned by prim::fork.
  auto reversed = b->nodes().reverse();
  for (auto it = reversed.begin(); it != reversed.end(); it++) {
    auto node = *it;
    if (node->kind() == prim::fork) {
      // Account for the case where the aten::wait call isn't present in
      // the current graph.
      node->output()->replaceAllUsesWith(future_remap.at(node->output()));
      it.destroyCurrent();
    } else if (node->kind() == aten::wait) {
      AT_ASSERT(node->inputs().size() == 1);
      AT_ASSERT(node->outputs().size() == 1);
      // If the future does not map to a prim::fork, it could be
      // returned from prim::rpc_async, which has side effect, so it shouldn't
      // be dead code eliminated.
      if (future_remap.count(node->input())) {
        node->output()->replaceAllUsesWith(future_remap.at(node->input()));
        it.destroyCurrent();
      }
    }
  }

  // Recursively inline fork/wait.
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    for (auto sub_b : node->blocks()) {
      InlineForkWait(sub_b, future_remap);
    }
  }
}

void InlineForkWait(const std::shared_ptr<Graph>& graph) {
  std::unordered_map<Value*, Value*> future_remap;
  InlineForkWait(graph->block(), future_remap);
  GRAPH_DUMP("After InlineForkWait: ", graph);
}

} // namespace torch::jit
