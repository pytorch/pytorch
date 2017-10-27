#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

// The intent for this optimization pass is to catch all of the small, easy to
// catch peephole optimizations you might be interested in doing.
//
// Right now, it does:
//    - Redundant 'expand' elimination
//
// TODO: Decide what kind of fixed point strategy we will have
void PeepholeOptimize(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    auto* n = *it;

    if (n->kind() == kexpand) {
      if (n->is(ksize) == n->input()->type()->expect<TensorType>()->sizes()) {
        // Attractive but wrong way to do this:
        // n->replaceAllUsesWith(n->input());
        auto* out = n->outputs().at(0);
        out->replaceAllUsesWith(n->input());
        out->destroy();
        it.destroyCurrent();
        continue;
      }
    }
  }
}

}}
