#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

namespace {

bool allSelectsUnused(Node* node) {
  for (auto & use : node->uses()) {
    if (use.user->uses().size() != 0)
      return false;
  }
  return true;
}

} // anonymous namespace

void EliminateDeadCode(std::shared_ptr<Graph>& graph) {
  auto nodes = graph->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    if (node->kind() == kSelect)
      continue;

    if (node->hasMultipleOutputs() && allSelectsUnused(node)) {
      auto uses = node->uses(); // A copy so we don't modify it within a range loop
      for (auto & use : uses)
        use.user->destroy();
      it.destroyCurrent();
    } else if (node->uses().size() == 0) {
      it.destroyCurrent();
    }
  }
}

}}
