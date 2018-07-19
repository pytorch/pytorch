#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

static void RemoveExpands(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      RemoveExpands(sub);
    if (it->kind() == aten::expand && it->hasAttribute(attr::implicit) && it->i(attr::implicit)) {
      it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    }
  }
}

void RemoveExpands(const std::shared_ptr<Graph>& graph) {
  RemoveExpands(graph->block());
}


}}
