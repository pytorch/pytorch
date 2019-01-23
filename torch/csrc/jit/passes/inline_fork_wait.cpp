#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

void InlineForkWait(Block* b, std::unordered_map<Value*, Value*>& future_remap) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::fork) {
      WithInsertPoint insert_guard(n);
      auto graph = b->owningGraph();
      auto subgraph = n->g(attr::Subgraph);
      // Map subgraph values -> this graph values
      std::unordered_map<Value*, Value*> value_remap;
      JIT_ASSERT(n->inputs().size() == subgraph->inputs().size());
      for (size_t i = 0; i < n->inputs().size(); ++i) {
        value_remap[subgraph->inputs()[i]] = n->input(i);
      }
      auto remap_fn = [&](Value* sub_val) -> Value* {
        return value_remap.at(sub_val);
      };
      for (auto sub_n : subgraph->nodes()) {
        auto cloned_node = graph->insertNode(graph->createClone(sub_n, remap_fn));
        JIT_ASSERT(sub_n->outputs().size() == cloned_node->outputs().size());
        for (size_t i = 0; i < sub_n->outputs().size(); ++i) {
          value_remap[sub_n->output(i)] = cloned_node->output(i);
        }
      }

      JIT_ASSERT(n->outputs().size() == 1);
      JIT_ASSERT(subgraph->outputs().size() == 1);
      JIT_ASSERT(value_remap.count(subgraph->outputs()[0]) > 0);

      future_remap[n->output()] = value_remap[subgraph->outputs()[0]];
    } else if (n->kind() == Symbol::fromQualString("aten::wait")) {
      JIT_ASSERT(n->inputs().size() == 1);
      JIT_ASSERT(n->outputs().size() == 1);
      JIT_ASSERT(future_remap.count(n->input()) > 0);
      n->output()->replaceAllUsesWith(future_remap[n->input()]);
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
