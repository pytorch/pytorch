#include "torch/csrc/jit/passes/erase_detach.h"
#include "torch/csrc/jit/constants.h"

namespace torch { namespace jit {

static void EraseDetach(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* n = *it;
    it++;

    for (auto sub : it->blocks()) {
      EraseDetach(sub);
    }

    if (n->matches("aten::detach(Tensor self) -> Tensor")) {
      n->output()->replaceAllUsesWith(n->input());
    }
  }
}

void EraseDetach(const std::shared_ptr<Graph>& graph) {
  EraseDetach(graph->block());
}

}}
