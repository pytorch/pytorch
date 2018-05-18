#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

bool hasSideEffects(Node * node) {
  return node->kind() == prim::Print ||
         node->kind() == prim::PythonOp ||
         node->kind() == prim::CppOp ||
         std::any_of(node->blocks().begin(), node->blocks().end(),
                     [](Block *b) {
                       return std::any_of(b->nodes().begin(), b->nodes().end(), hasSideEffects);
                     });
}

void EliminateDeadCode(std::shared_ptr<Graph>& graph) {
  EliminateDeadCode(graph->block());
}

void EliminateDeadCode(Block *block, bool recurse) {
  auto nodes = block->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    if (recurse) {
      for (Block * block : node->blocks())
        EliminateDeadCode(block);
    }
    if (!node->hasUses() && !hasSideEffects(node))
      it.destroyCurrent();
  }
}

}} // namespace torch::jit
