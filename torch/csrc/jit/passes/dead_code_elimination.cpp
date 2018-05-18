#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

bool hasSideEffects(Node * node) {
  // FIXME: PythonOp and CppOp should be treated as having side effects as well!
  //        Unfortunately ONNX depends on them getting removed in this pass, so it's not
  //        a simple change.
  return node->kind() == prim::Print ||
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
