#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::shared_ptr<Graph>& graph) {
  EliminateDeadCode(graph->block());
}
bool hasSideEffects(Node * node) {
  return node->kind() == prim::Print || node->blocks().size() > 0;
}

void EliminateDeadCode(Block *block) {
  auto nodes = block->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    for (Block * block : node->blocks())
      EliminateDeadCode(block);
    if (!node->hasUses() && !hasSideEffects(node))
      it.destroyCurrent();
  }
}

}}
