#include "torch/csrc/jit/dead_code_elimination.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::unique_ptr<Graph>& graph) {
  auto nodes = graph->nodes().reverse();
  for(auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    if(node->uses().size() == 0) {
      it.destroyCurrent();
    }
  }
}

}}
