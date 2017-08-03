#include "torch/csrc/jit/dead_code_elimination.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::unique_ptr<Graph>& graph) {
  auto& nodes = graph->nodes();
  for (auto it = nodes.rbegin(); it != nodes.rend();) {
    Node *node = *it++;
    if (node->uses().size() == 0) {
      node->destroy();
    }
  }
}

}}
