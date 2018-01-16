#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::shared_ptr<Graph>& graph) {
  auto nodes = graph->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    if(!node->hasUses())
      it.destroyCurrent();
  }
}

}}
